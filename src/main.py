import pyrootutils

pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)
from src.utils import CustomLightningCLI  # noqa: E402

if __name__ == "__main__":
    cli = CustomLightningCLI()
