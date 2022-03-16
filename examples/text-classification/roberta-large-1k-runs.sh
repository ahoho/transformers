#!/bin/bash
#SBATCH --constraint=gpu-small
#SBATCH --array=0-8
#SBATCH --job-name=roberta-large-1k
#SBATCH --output=/workspace/transformers/examples/text-classification/sbatch-logs/roberta-large-1k-%A-%a.log
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

test ${SLURM_ARRAY_TASK_ID} -eq 0 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-05-epochs_10-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 1 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-05-epochs_20-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 2 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-05-epochs_30-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 3 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_10-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 4 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_20-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 5 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 6 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_10-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 7 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_False-shuffle_trees/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 8 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_False-shuffle_trees/config.yml