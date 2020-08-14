#! /bin/bash
export PYTHONPATH="../":"${PYTHONPATH}"

/home/alexanderh/miniconda3/envs/cgen-transformer/bin/python finetune.py \
--data_dir=../../../data/webnlg/linearized/complete-entities \
--learning_rate=0.001 \
--train_batch_size=128 \
--eval_batch_size=128 \
--output_dir=../../../outputs/t5-linearized_complete-default_gen_hp-beam_10-lr_0.001-bs_256 \
--max_source_length=176 \
--max_target_length=122 \
--val_max_target_length=122 \
--test_max_target_length=122 \
--num_beams 10 \
--val_check_interval=1 --n_val=200 \
--do_train --do_predict \
--model_name_or_path t5-small \
--task data-to-text \
--gpus=1 \
--new_tokens_fpath=../../../added_tokens.json \
--max_steps=1000 \
--gradient_accumulation_steps=2 \
--max_epochs=5