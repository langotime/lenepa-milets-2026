from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "pretrain"


def _compose(config_name: str, overrides: list[str]):
  with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
    return compose(config_name=config_name, overrides=overrides)


def test_core_imports():
  import configs  # noqa: F401
  from losses.sigreg import SIGReg  # noqa: F401
  from models import JEPA, NEPA, VisionTransformer  # noqa: F401


def test_ptbxl_lenepa_config_composes():
  cfg = _compose(
    "ViTXS_ptbxl",
    [
      "run_name=smoke",
      "+data.ptb-xl=/tmp/ptbxl.npy",
      "ptb_xl_data_dir=/tmp/ptbxl",
      "objective=nepa_sigreg",
      "ema_encoder=false",
      "pred_depth=0",
      "nepa_sigreg_scale=null",
      "nepa_sigreg_layers=[]",
      "nepa_sigreg_scale_time=20",
      "nepa_sigreg_time_layers=[0,8]",
      "offline_probe_layers_categorical=[0,1,2,3,4,5,6,7,8]",
      "offline_probe_layers_dense=[0,1,2,3,4,5,6,7,8]",
      "use_projector=true",
    ],
  )
  assert cfg.run_name == "smoke"
  assert cfg.use_nepa is True
  assert cfg.nepa_stability == "sigreg"
  assert cfg.nepa_sigreg_time_layers == [0, 8]


def test_diag_lenepa_config_composes():
  cfg = _compose(
    "ViTXS_aiono",
    [
      "run_name=smoke_diag",
      "dataset=aiono_basic_components_imbalanced",
      "offline_probe_dataset=same_as_train",
      "objective=nepa_sigreg",
      "ema_encoder=false",
      "pred_depth=0",
      "nepa_sigreg_scale=null",
      "nepa_sigreg_layers=[]",
      "nepa_sigreg_scale_time=20",
      "nepa_sigreg_time_layers=[0,8]",
    ],
  )
  assert cfg.data_backend == "aiono"
  assert cfg.probe_source == "aiono"
  assert cfg.aiono.benchmark_family == "aiono_basic_components"
