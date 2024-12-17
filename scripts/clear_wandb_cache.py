"""
This script is used to delete unused offline wandb runs and model checkpoints.
"""

import os
import shutil

import pyrootutils
import wandb

pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=False, pythonpath=True, cwd=True)

WANDB_PROJECT = "mnist"
MODEL_CHECKPOINT_PATH = os.path.join("logs", "mnist")
LOG_PATH = os.path.join("logs", "wandb")
WHITE_LIST = [
    "<RUN_ID>",
]

if __name__ == "__main__":
    # get run logs
    run_id_to_log_path_map = {}
    if os.path.exists(LOG_PATH):
        for dir in os.listdir(LOG_PATH):
            log_path = os.path.join(LOG_PATH, dir)
            if os.path.isdir(log_path) and not os.path.islink(log_path):
                version = dir.split("-")[-1]
                run_id_to_log_path_map[version] = log_path

    # get model checkpoints
    run_id_to_model_checkpoint_path_map = {}
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        for version in os.listdir(MODEL_CHECKPOINT_PATH):
            model_checkpoint_path = os.path.join(MODEL_CHECKPOINT_PATH, version)
            if os.path.isdir(model_checkpoint_path):
                run_id_to_model_checkpoint_path_map[version] = model_checkpoint_path

    # get online runs
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT)
    online_run_id_to_name_map = {}
    for run in runs:
        online_run_id_to_name_map[run.id] = run.name

    # delete offline runs
    keep_count = 0
    delete_count = 0
    for run_id, log_path in run_id_to_log_path_map.items():
        if run_id not in online_run_id_to_name_map:
            # print(f"Deleting {run_id}")
            shutil.rmtree(log_path)
            delete_count += 1
        else:
            # print(f"Keeping {online_run_id_to_name_map[run_id]}({run_id})")
            keep_count += 1
    print(f"{keep_count} runs kept and {delete_count} runs deleted")

    # delete offline model checkpoints
    keep_count = 0
    delete_count = 0
    for run_id, model_checkpoint_path in run_id_to_model_checkpoint_path_map.items():
        if run_id not in online_run_id_to_name_map:
            if run_id in WHITE_LIST:
                continue
            # print(f"Deleting {run_id}")
            shutil.rmtree(model_checkpoint_path)
            delete_count += 1
        else:
            # print(f"Keeping {online_run_id_to_name_map[run_id]}({run_id})")
            keep_count += 1
    print(f"{keep_count} model checkpoints kept and {delete_count} model checkpoints deleted")
