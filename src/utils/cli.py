import os
import warnings
from typing import Any, Dict, Optional, Type

from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerTypeUnion,
    ReduceLROnPlateau,
    SaveConfigCallback,
)
from pytorch_lightning.loggers import Logger, WandbLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR


class WandbSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        log_dir = trainer.log_dir  # this broadcasts the directory
        if trainer.logger is not None and trainer.logger.name is not None and trainer.logger.version is not None:
            log_dir = os.path.join(log_dir, trainer.logger.name, str(trainer.logger.version))
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True
            # save optimizer and lr scheduler config
            for _logger in trainer.loggers:
                if isinstance(_logger, Logger):
                    config = {}
                    if "optimizer" in self.config:
                        config["optimizer"] = {
                            k.replace("init_args.", ""): v for k, v in dict(self.config["optimizer"]).items()
                        }
                    if "lr_scheduler" in self.config:
                        config["lr_scheduler"] = {
                            k.replace("init_args.", ""): v for k, v in dict(self.config["lr_scheduler"]).items()
                        }
                    _logger.log_hyperparams(config)

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class CustomLightningCLI(LightningCLI):
    def __init__(
        self,
        save_config_callback: Optional[Type[SaveConfigCallback]] = WandbSaveConfigCallback,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        new_parser_kwargs = {
            sub_command: dict(default_config_files=[os.path.join("configs", "default.yaml")])
            for sub_command in ["fit", "validate", "test", "predict"]
        }
        new_parser_kwargs.update(parser_kwargs or {})
        super().__init__(save_config_callback=save_config_callback, parser_kwargs=new_parser_kwargs, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--ignore_warnings", default=False, type=bool, help="Ignore warnings")
        parser.add_argument("--git_commit_before_fit", default=False, type=bool, help="Git commit before training")
        parser.add_argument(
            "--test_after_fit", default=False, type=bool, help="Run test on the best checkpoint after training"
        )

    def before_instantiate_classes(self) -> None:
        if self.config[self.subcommand].get("ignore_warnings"):
            warnings.filterwarnings("ignore")

    def before_fit(self) -> None:
        if self.config.fit.get("git_commit_before_fit") and not os.environ.get("DEBUG", False):
            logger = self.trainer.logger
            if isinstance(logger, WandbLogger):
                version = getattr(logger, "version")
                name = getattr(logger, "_name")
                message = "Commit Message"
                if name and version:
                    message = f"{name}_{version}"
                elif name:
                    message = name
                elif version:
                    message = version
                os.system(f'git commit -am "{message}"')

    def after_fit(self) -> None:
        if self.config.fit.get("test_after_fit") and not os.environ.get("DEBUG", False):
            self._run_subcommand("test")

    def before_test(self) -> None:
        if self.trainer.checkpoint_callback and self.trainer.checkpoint_callback.best_model_path:
            tested_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        elif self.config_init[self.config_init["subcommand"]]["ckpt_path"]:
            return
        else:
            tested_ckpt_path = None
        self.config_init[self.config_init["subcommand"]]["ckpt_path"] = tested_ckpt_path

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v
            for k, v in self.config_init[self.config_init["subcommand"]].items()
            if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        if isinstance(lr_scheduler, (OneCycleLR, CyclicLR)):
            # CyclicLR and OneCycleLR are step-based schedulers, where the default interval is "epoch".
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}
        return [optimizer], [lr_scheduler]
