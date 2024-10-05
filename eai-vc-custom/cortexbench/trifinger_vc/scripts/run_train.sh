GPU=$1
METHOD=$2
ENTITY=$3
PROJECT=$4

CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=12345 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=23451 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=34512 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=45123 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=51234 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=reach_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT

CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=12345 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=23451 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=34512 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=45123 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
CUDA_VISIBLE_DEVICES=$GPU SLURM_STEP_GPUS=0 python bc_experiments/train_bc.py seed=51234 algo.pretrained_rep=${METHOD} algo.freeze_pretrained_rep=true rep_to_policy=none task=move_cube wandb.entity=$ENTITY wandb.mode=online wandb.project=$PROJECT
