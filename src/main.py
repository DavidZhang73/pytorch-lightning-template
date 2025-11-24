import os

import autorootcwd  # noqa: F401

from src.utils import CustomLightningCLI  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("DEBUG", False):
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    cli = CustomLightningCLI()
