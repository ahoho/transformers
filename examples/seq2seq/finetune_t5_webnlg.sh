#! /bin/bash
export PYTHONPATH="../":"${PYTHONPATH}"

/home/alexanderh/miniconda3/envs/cgen-transformer/bin/python finetune.py \
--data_dir=../../../data/webnlg/linearized \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=../../../outputs/t5-linearized-5_epochs-eos_token \
--max_source_length=512 \
--val_check_interval=1.0 --n_val=200 \
--do_train --do_predict \
--model_name_or_path t5-small \
--task translation \
--fp16 \
--num_train_epochs 5