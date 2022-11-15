import os
import json


# Train baseline model
dense_train = """CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port='2004' ./question_answering/run_qa.py \
--model_name_or_path bert-large-uncased-whole-word-masking --dataset_name squad \
--overwrite_output_dir --do_train --per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--learning_rate 3e-5 --save_steps 10000 --num_train_epochs 2 --max_seq_length 384 \
--doc_stride 128 --output_dir ./ckpt/bert_large_squad/ --seed 0"""

os.system(dense_train)

dsp_train = """CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port='2004' ./question_answering/run_qa.py \
--model_name_or_path bert-large-uncased-whole-word-masking --dataset_name squad \
--overwrite_output_dir --do_train --per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--learning_rate 3e-5 --save_steps 10000 --num_train_epochs 2 --max_seq_length 384 \
--doc_stride 128 --output_dir ./ckpt/bert_large_squad_dsp/ --dsp --seed 0"""

os.system(dsp_train)

# evaluate and collect the results
dense_eval = """CUDA_VISIBLE_DEVICES=0 python ./question_answering/run_qa.py \
--model_name_or_path ./ckpt/bert_large_squad/ --dataset_name squad \
--overwrite_output_dir --do_eval --per_device_eval_batch_size 24 \
--max_seq_length 384 --doc_stride 128 --output_dir ./ckpt/bert_large_squad_bf16_eval/ --bf16"""

os.system(dense_eval)

dsp_eval = """CUDA_VISIBLE_DEVICES=0 python ./question_answering/run_qa.py \
--model_name_or_path ./ckpt/bert_large_squad_dsp/ --dataset_name squad \
--overwrite_output_dir --do_eval --per_device_eval_batch_size 24 \
--max_seq_length 384 --doc_stride 128 --output_dir ./ckpt/bert_large_squad_dsp_bf16_eval/ --bf16"""

os.system(dsp_eval)


# collect result 
with open('./ckpt/bert_large_squad_bf16_eval/eval_results.json') as f:
    dense_f1 = json.load(f)["eval_f1"]

with open('./ckpt/bert_large_squad_dsp_bf16_eval/eval_results.json') as f:
    dsp_f1 = json.load(f)["eval_f1"]

print("F1 score on BERT-large SQuAD v1.1")
print("Transformer: %.2f, DFSS 2:4: %.2f" % (dense_f1, dsp_f1))