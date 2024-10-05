GPU=$1
ENV=$2
YAML=$3
METHOD=$4
ENTITY=$5

CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name Adroit_BC_config.yaml env=$ENV embedding=$YAML seed=1 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name Adroit_BC_config.yaml env=$ENV embedding=$YAML seed=2 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name Adroit_BC_config.yaml env=$ENV embedding=$YAML seed=3 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name Adroit_BC_config.yaml env=$ENV embedding=$YAML seed=4 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=$GPU python hydra_launcher.py --config-name Adroit_BC_config.yaml env=$ENV embedding=$YAML seed=5 wandb.project=cortex_$METHOD wandb.entity=$ENTITY wandb.mode=online