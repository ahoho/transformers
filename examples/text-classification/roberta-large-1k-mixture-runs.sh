#!/bin/bash
#SBATCH --constraint=gpu-small
#SBATCH --array=0-22
#SBATCH --job-name=roberta-large-1k-mixture
#SBATCH --output=/workspace/transformers/examples/text-classification/sbatch-logs/roberta-large-1k-mixture-%A-%a.log
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

test ${SLURM_ARRAY_TASK_ID} -eq 0 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.25-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 1 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.25-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 2 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.25-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 3 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.5-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 4 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.5-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 5 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.5-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 6 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.75-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 7 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.75-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 8 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_1e-05-epochs_30-text_0.75-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 9 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.25-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 10 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.25-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 11 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.25-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 12 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_0.5-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 13 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_0.5-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 14 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.5-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 15 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.5-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 16 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.5-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 17 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_0.75-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 18 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_0.75-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 19 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_20-text_0.75-shuffle_0.75/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 20 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.75-shuffle_0.25/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 21 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.75-shuffle_0.5/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 22 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/1k/roberta-large/lr_5e-06-epochs_30-text_0.75-shuffle_0.75/config.yml