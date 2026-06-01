"""Invertible flow components (vendored from ml-tidbits).

This module vendors the minimal subset of `EmbedModels.py` needed for Plan 113:
  - `C_ACN`
  - `C_PermutationLayer`
  - `C_AffineCouplingLayer`
  - `C_InvertibleFlow`

Math is kept identical to the source implementation; only style/typing/comments
are adapted to match this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
  "C_ACN",
  "C_PermutationLayer",
  "C_AffineCouplingLayer",
  "C_InvertibleFlow",
]


class C_ACN(nn.Module):
  """Auto-compressing network (ACN): residual MLP with additive blocks.

  Given input features $x \\in \\mathbb{R}^{D_{in}}$, ACN computes:
  - $a_0 = W_{in} x + b_{in}$
  - For each block $k$: $a_k = W_k \\mathrm{ELU}(a_{k-1}) + b_k$ and accumulate
    $r = r + a_k$ (additive residual across blocks)
  - Output: $W_{out} \\mathrm{ELU}(r) + b_{out}$

  Shapes:
    - Input: `x[..., in_dim]`
    - Output: `y[..., out_dim]`
  """

  def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, n_blocks: int = 2) -> None:
    super().__init__()
    if in_dim < 1:
      raise ValueError(f"in_dim must be >= 1, got {in_dim}")
    if out_dim < 1:
      raise ValueError(f"out_dim must be >= 1, got {out_dim}")
    if hidden_dim < 1:
      raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
    if n_blocks < 1:
      raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

    self.in_proj = nn.Linear(int(in_dim), int(hidden_dim))
    self.blocks = nn.ModuleList([nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(n_blocks))])
    self.out_proj = nn.Linear(int(hidden_dim), int(out_dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [..., in_dim]
    a = self.in_proj(x)  # [..., hidden_dim]
    res = a  # [..., hidden_dim]
    for lin in self.blocks:
      a = lin(F.elu(a))  # [..., hidden_dim]
      res = res + a  # [..., hidden_dim]
    return self.out_proj(F.elu(res))  # [..., out_dim]


class C_PermutationLayer(nn.Module):
  """Deterministic permutation of the last dimension with an exact inverse.

  Shapes:
    - Input:  `x[..., dim]`
    - Output: `y[..., dim]`
  """

  def __init__(self, dim: int, perm: torch.Tensor) -> None:
    super().__init__()
    dim = int(dim)
    if dim < 1:
      raise ValueError("dim must be >= 1")
    if perm.ndim != 1 or int(perm.numel()) != dim:
      raise ValueError("perm must have shape (dim,)")

    perm = perm.to(dtype=torch.int64).contiguous()  # [dim]
    inv = torch.empty_like(perm)  # [dim]
    inv[perm] = torch.arange(dim, dtype=torch.int64)  # [dim]

    self.dim = dim
    self.register_buffer("perm", perm)  # [dim]
    self.register_buffer("inv_perm", inv)  # [dim]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [..., dim]
    if x.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
    return torch.index_select(x, -1, self.perm)  # [..., dim]

  @torch.jit.export
  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    # y: [..., dim]
    if y.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
    return torch.index_select(y, -1, self.inv_perm)  # [..., dim]


class C_AffineCouplingLayer(nn.Module):
  """RealNVP-style affine coupling layer with exact inverse.

  This layer applies an affine transform to the "transformed" subset of features:
    y_trans = x_trans * exp(s(x_pass)) + t(x_pass)
  where `(s, t)` are produced by a conditioner network that sees `x_pass`.

  Numerical stability:
    - log-scale is bounded: `s = tanh(s_raw) * s_max`.

  Shapes:
    - Input:  `x[..., D]`
    - Output: `y[..., D]`
  """

  def __init__(
    self,
    dim: int,
    mask: torch.Tensor,
    hidden_dim: int = 128,
    n_blocks: int = 2,
    s_max: float = 2.0,
  ) -> None:
    super().__init__()
    dim = int(dim)
    if dim < 1:
      raise ValueError("dim must be >= 1")
    if hidden_dim < 1:
      raise ValueError("hidden_dim must be >= 1")
    if n_blocks < 1:
      raise ValueError("n_blocks must be >= 1")
    if s_max <= 0:
      raise ValueError("s_max must be > 0")

    self.dim = dim
    self.s_max = float(s_max)

    if mask.ndim != 1 or int(mask.numel()) != dim:
      raise ValueError("mask must have shape (dim,)")

    m = mask.to(dtype=torch.float32).contiguous()  # [D]
    self.register_buffer("mask", m)  # [D]
    self.register_buffer("inv_mask", 1.0 - m)  # [D]

    pass_idx = torch.nonzero(m >= 0.5, as_tuple=False).flatten().to(dtype=torch.int64)  # [D_pass]
    trans_idx = torch.nonzero(m < 0.5, as_tuple=False).flatten().to(dtype=torch.int64)  # [D_trans]

    self.register_buffer("pass_idx", pass_idx)  # [D_pass]
    self.register_buffer("trans_idx", trans_idx)  # [D_trans]
    self.pass_dim = int(pass_idx.numel())
    self.trans_dim = int(trans_idx.numel())

    if self.pass_dim < 1 or self.trans_dim < 1:
      self.net: C_ACN | None = None
      self._contig = True
      self._pass_first = True
      return

    # Fast path detection: contiguous split either [pass | trans] or [trans | pass].
    self._contig = False
    self._pass_first = True
    with torch.no_grad():
      a = torch.arange(self.dim, dtype=torch.int64)  # [D]
      if torch.equal(pass_idx, a[: self.pass_dim]) and torch.equal(trans_idx, a[self.pass_dim :]):
        self._contig = True
        self._pass_first = True
      elif torch.equal(trans_idx, a[: self.trans_dim]) and torch.equal(pass_idx, a[self.trans_dim :]):
        self._contig = True
        self._pass_first = False

    self.net = C_ACN(self.pass_dim, 2 * self.trans_dim, hidden_dim=int(hidden_dim), n_blocks=int(n_blocks))

    # Identity-ish init: start close to identity for stability (s=t=0).
    with torch.no_grad():
      nn.init.zeros_(self.net.out_proj.weight)
      nn.init.zeros_(self.net.out_proj.bias)

  def _ST(self, x_pass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x_pass: [..., D_pass]
    if self.net is None:
      raise RuntimeError("Coupling layer conditioner is not initialized (degenerate mask).")
    st = self.net(x_pass)  # [..., 2*D_trans]
    s_raw, t = st.chunk(2, dim=-1)  # [..., D_trans], [..., D_trans]
    s = torch.tanh(s_raw) * self.s_max  # [..., D_trans]
    return s, t

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [..., D]
    if x.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
    if self.net is None:
      return x

    if self._contig:
      if self._pass_first:
        x_pass = x[..., : self.pass_dim]  # [..., D_pass]
        x_trans = x[..., self.pass_dim :]  # [..., D_trans]
        s, t = self._ST(x_pass)  # [..., D_trans], [..., D_trans]
        y_trans = x_trans * torch.exp(s) + t  # [..., D_trans]
        return torch.cat((x_pass, y_trans), dim=-1)  # [..., D]
      x_trans = x[..., : self.trans_dim]  # [..., D_trans]
      x_pass = x[..., self.trans_dim :]  # [..., D_pass]
      s, t = self._ST(x_pass)  # [..., D_trans], [..., D_trans]
      y_trans = x_trans * torch.exp(s) + t  # [..., D_trans]
      return torch.cat((y_trans, x_pass), dim=-1)  # [..., D]

    x_pass = torch.index_select(x, -1, self.pass_idx)  # [..., D_pass]
    x_trans = torch.index_select(x, -1, self.trans_idx)  # [..., D_trans]
    s, t = self._ST(x_pass)  # [..., D_trans], [..., D_trans]
    y_trans = x_trans * torch.exp(s) + t  # [..., D_trans]

    y = torch.empty_like(x)  # [..., D]
    y.index_copy_(-1, self.pass_idx, x_pass)
    y.index_copy_(-1, self.trans_idx, y_trans)
    return y

  @torch.jit.export
  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    # y: [..., D]
    if y.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
    if self.net is None:
      return y

    if self._contig:
      if self._pass_first:
        y_pass = y[..., : self.pass_dim]  # [..., D_pass]
        y_trans = y[..., self.pass_dim :]  # [..., D_trans]
        s, t = self._ST(y_pass)  # [..., D_trans], [..., D_trans]
        x_trans = (y_trans - t) * torch.exp(-s)  # [..., D_trans]
        return torch.cat((y_pass, x_trans), dim=-1)  # [..., D]
      y_trans = y[..., : self.trans_dim]  # [..., D_trans]
      y_pass = y[..., self.trans_dim :]  # [..., D_pass]
      s, t = self._ST(y_pass)  # [..., D_trans], [..., D_trans]
      x_trans = (y_trans - t) * torch.exp(-s)  # [..., D_trans]
      return torch.cat((x_trans, y_pass), dim=-1)  # [..., D]

    y_pass = torch.index_select(y, -1, self.pass_idx)  # [..., D_pass]
    y_trans = torch.index_select(y, -1, self.trans_idx)  # [..., D_trans]
    s, t = self._ST(y_pass)  # [..., D_trans], [..., D_trans]
    x_trans = (y_trans - t) * torch.exp(-s)  # [..., D_trans]

    x = torch.empty_like(y)  # [..., D]
    x.index_copy_(-1, self.pass_idx, y_pass)
    x.index_copy_(-1, self.trans_idx, x_trans)
    return x


class C_InvertibleFlow(nn.Module):
  """Composable invertible flow: coupling layers with optional permutations.

  The flow is a sequence of operators with an exact inverse:
    - optional permutation layers (indexing, exact inverse)
    - affine coupling layers (RealNVP-style, exact inverse)

  Shapes:
    - Input:  `x[..., dim]`
    - Output: `y[..., dim]`
  """

  def __init__(
    self,
    dim: int,
    n_layers: int = 6,
    hidden_dim: int = 128,
    n_blocks: int = 2,
    s_max: float = 2.0,
    mask_mode: str = "alternating",
    *,
    permute_mode: str = "per_pair",
    permute_seed: int = 1337,
  ) -> None:
    super().__init__()
    dim = int(dim)
    if dim < 1:
      raise ValueError("dim must be >= 1")
    if n_layers < 0:
      raise ValueError("n_layers must be >= 0")
    if hidden_dim < 1:
      raise ValueError("hidden_dim must be >= 1")
    if n_blocks < 1:
      raise ValueError("n_blocks must be >= 1")
    if s_max <= 0:
      raise ValueError("s_max must be > 0")

    self.dim = dim
    self.n_layers = int(n_layers)
    self.hidden_dim = int(hidden_dim)
    self.n_blocks = int(n_blocks)
    self.s_max = float(s_max)

    if mask_mode not in ("alternating", "half"):
      raise ValueError("mask_mode must be 'alternating' or 'half'")
    self.mask_mode = str(mask_mode)

    if permute_mode not in ("none", "per_layer", "per_pair"):
      raise ValueError("permute_mode must be 'none', 'per_layer', or 'per_pair'")
    self.permute_mode = str(permute_mode)
    self.permute_seed = int(permute_seed) & 0x7FFFFFFF

    ops: list[nn.Module] = []
    if self.n_layers > 0 and self.dim >= 2:
      idx = torch.arange(self.dim, dtype=torch.int64)  # [D]
      d1 = self.dim // 2

      # Base half masks in current coordinate order.
      mask0 = torch.zeros((self.dim,), dtype=torch.float32)  # [D]
      mask0[:d1] = 1.0
      mask1 = 1.0 - mask0  # [D]

      if self.permute_mode != "none":
        # Deterministic permutation stream.
        gen = torch.Generator()
        gen.manual_seed(self.permute_seed)

        for i in range(self.n_layers):
          if (self.permute_mode == "per_layer") or (self.permute_mode == "per_pair" and ((int(i) & 1) == 0)):
            perm = torch.randperm(self.dim, generator=gen)  # [D]
            ops.append(C_PermutationLayer(self.dim, perm))

          m = mask0 if ((int(i) & 1) == 0) else mask1
          ops.append(
            C_AffineCouplingLayer(
              self.dim,
              m,
              hidden_dim=self.hidden_dim,
              n_blocks=self.n_blocks,
              s_max=self.s_max,
            )
          )
      else:
        for i in range(self.n_layers):
          if self.mask_mode == "alternating":
            m = ((idx + int(i)) & 1).to(dtype=torch.float32)  # [D]
          else:
            m = mask0 if ((int(i) & 1) == 0) else mask1  # [D]
          ops.append(
            C_AffineCouplingLayer(
              self.dim,
              m,
              hidden_dim=self.hidden_dim,
              n_blocks=self.n_blocks,
              s_max=self.s_max,
            )
          )

    self.ops = nn.ModuleList(ops)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [..., dim]
    if x.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
    y = x
    for op in self.ops:
      y = op(y)
    return y  # [..., dim]

  @torch.jit.export
  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    # y: [..., dim]
    if y.shape[-1] != self.dim:
      raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
    x = y
    for op in self.ops[::-1]:
      x = op.inverse(x)
    return x  # [..., dim]
