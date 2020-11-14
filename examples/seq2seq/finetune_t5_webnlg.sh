#! /bin/bash
export PYTHONPATH="../":"${PYTHONPATH}"

/home/alexanderh/miniconda3/envs/cgen-transformer/bin/python finetune.py \
--data_dir=../../../data/webnlg/linearized/complete-entities-demarcated \
--learning_rate=0.001 \
--train_batch_size=32 \
--eval_batch_size=32 \
--output_dir=../../../models/webnlg/linearized-t5/baseline-large-lr_0.001 \
--max_source_length=176 \
--max_target_length=122 \
--val_max_target_length=122 \
--test_max_target_length=122 \
--num_beams 10 \
--val_check_interval=1 \
--do_train \
--model_name_or_path t5-large \
--task data-to-text \
--gpus=1 \
--adafactor \
--new_tokens_fpath=../../../added_tokens.json \
--max_steps=5000 \
--gradient_accumulation_steps=8 \
--shuffle_graph_components \
--seed 11235
#--shuffle_graph_subcomponents