GPU=$1
ENV=$2
YAML=$3
METHOD=$4
ENTITY=$5

CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name DMC_BC_config.yaml env=$ENV embedding=$YAML seed=12345 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name DMC_BC_config.yaml env=$ENV embedding=$YAML seed=23451 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name DMC_BC_config.yaml env=$ENV embedding=$YAML seed=34512 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name DMC_BC_config.yaml env=$ENV embedding=$YAML seed=45123 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name DMC_BC_config.yaml env=$ENV embedding=$YAML seed=51234 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online