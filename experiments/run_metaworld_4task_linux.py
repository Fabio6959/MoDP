import os
import subprocess
import sys
from datetime import datetime

date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f"_{os.getpid()}"
script_name = os.path.basename(__file__).replace(".py", "")
pretrained = "pretrained/hpt-small"
pretrained_cmd = sys.argv[2] if len(sys.argv) > 2 else ""
postfix = "_hpt_baseline_metaworld_4task_freezeFalse"
print(f"RUNNING {script_name}!")

env = os.environ.copy()
env["HYDRA_FULL_ERROR"] = "1"
env["CUDA_VISIBLE_DEVICES"] = "0"
env["WANDB_MODE"] = "offline"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "env", "mujoco")
pythonpath = env.get("PYTHONPATH", "")
if pythonpath:
    env["PYTHONPATH"] = f"{project_root}:{env_path}:{pythonpath}"
else:
    env["PYTHONPATH"] = f"{project_root}:{env_path}"

train_cmd = [
    sys.executable, "-m", "hpt.run",
    f"script_name={script_name}",
    f"train.pretrained_dir={pretrained}",
    f"dataset.episode_cnt=200",
    f"domains=metaworld_4task",
    f"+tasks=metaworld_4task_linux",
    f"train.freeze_trunk=False",
    f"dataset.regenerate=False",
    f"output_dir=output/{date_str}{postfix}",
]

print("Executing training command:")
print(" ".join(train_cmd))
result = subprocess.run(train_cmd, env=env)
print(f"Training finished with return code: {result.returncode}")

output_dir = f"output/{date_str}{postfix}"
eval_cmd = [
    sys.executable, "-m", "hpt.run_eval",
    f"train.pretrained_dir={output_dir}",
    f"domains=metaworld_4task",
    f"+tasks=metaworld_4task_linux",
    f"seed=42",
]

print("Executing evaluation command:")
print(" ".join(eval_cmd))
subprocess.run(eval_cmd, env=env)
