# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--data_dir=$CNN_DIR \
--learning_rate=3e-5 \
--train_batch_size=$BS \
--eval_batch_size=$BS \
--output_dir=$OUTPUT_DIR \
--max_source_length=512 \
--val_check_interval=0.1 \
--do_train --do_predict \
 $@
