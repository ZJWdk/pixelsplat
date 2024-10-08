from dataclasses import dataclass

import debugpy

debugpy.listen(("localhost", 5678))  # 监听本地端口
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # 等待 VSCode 连接
print("Debugger attached, continuing execution...")

import hydra
import torch
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from omegaconf import DictConfig

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset import DatasetCfg
    from src.dataset.data_module import DataLoaderCfg, DataModule
    from src.evaluation.evaluation_index_generator import (
        EvaluationIndexGenerator,
        EvaluationIndexGeneratorCfg,
    )
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    index_generator: EvaluationIndexGeneratorCfg
    seed: int


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="generate_extreme_identity_index",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices="auto", strategy="auto")
    data_module = DataModule(cfg.dataset, cfg.data_loader, None)
    evaluation_index_generator = EvaluationIndexGenerator(cfg.index_generator)
    trainer.test(evaluation_index_generator, datamodule=data_module)
    evaluation_index_generator.save_index()


if __name__ == "__main__":
    train()
