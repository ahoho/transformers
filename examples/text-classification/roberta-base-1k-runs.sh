#!/bin/bash
#SBATCH --constraint=gpu-small
#SBATCH --array=0-49
#SBATCH --job-name=roberta-base-1k
#SBATCH --output=/workspace/transformers/examples/text-classification/sbatch-logs/roberta-base-1k-%A-%a.log
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

test ${SLURM_ARRAY_TASK_ID} -eq 0 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_5-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 1 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_10-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 2 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_15-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 3 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_20-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 4 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_30-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 5 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_5-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 6 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_10-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 7 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_15-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 8 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_20-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 9 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.001-epochs_30-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 10 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_5-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 11 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_10-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 12 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_15-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 13 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_20-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 14 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_30-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 15 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_5-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 16 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_10-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 17 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_15-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 18 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_20-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 19 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_0.0001-epochs_30-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 20 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_5-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 21 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_10-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 22 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_15-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 23 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_20-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 24 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_30-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 25 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_5-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 26 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_10-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 27 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_15-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 28 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_20-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 29 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-05-epochs_30-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 30 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_5-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 31 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_10-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 32 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_15-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 33 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_20-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 34 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_30-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 35 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_5-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 36 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_10-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 37 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_15-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 38 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_20-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 39 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_1e-05-epochs_30-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 40 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_5-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 41 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_10-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 42 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_15-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 43 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_20-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 44 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_30-text_True/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 45 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_5-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 46 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_10-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 47 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_15-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 48 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_20-text_False/config.yml
test ${SLURM_ARRAY_TASK_ID} -eq 49 && /workspace/.conda/envs/transformers4/bin/python run_glue.py /workspace/transformers/examples/text-classification/outputs/lr_5e-06-epochs_30-text_False/config.yml