import os
import warnings
from typing import Any

from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger


class WandbSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        log_dir = trainer.log_dir  # this broadcasts the directory
        if trainer.logger is not None and trainer.logger.name is not None and trainer.logger.version is not None:
            log_dir = os.path.join(log_dir, trainer.logger.name, trainer.logger.version)
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
                    " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
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

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class CustomLightningCLI(LightningCLI):
    def __init__(
        self,
        save_config_callback: type[SaveConfigCallback] = WandbSaveConfigCallback,
        parser_kwargs: dict[str, Any] = None,
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
        if self.config.fit.get("git_commit_before_fit"):
            logger = self.trainer.logger
            if isinstance(logger, WandbLogger):
                version = getattr(logger, "version")
                name = getattr(logger, "name")
                message = "Commit Message"
                if name and version:
                    message = f"{name}_{version}"
                elif name:
                    message = name
                elif version:
                    message = version
                os.system(f'git commit -am "{message}"')

    def after_fit(self) -> None:
        if self.config.fit.get("test_after_fit"):
            self._run_subcommand("test")

    def after_test(self) -> None:
        if self.trainer.ckpt_path:
            tested_ckpt_path = self.trainer.ckpt_path
        elif self.trainer.checkpoint_callback and self.trainer.checkpoint_callback.best_model_path:
            tested_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        else:
            tested_ckpt_path = None
        print("Tested on checkpoint:", tested_ckpt_path)

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
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
