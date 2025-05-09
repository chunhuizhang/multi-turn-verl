#! /bin/bash

python tom_eval/tom_eval.py \
    --ckpt_path ./checkpoints/tom_sft_Qwen2.5-0.5B-Instruct/global_step_24 \
    --data_path ./tom_eval/tom_eval_datasets.csv \
    --max_model_len 2048 \
    --max_tokens 20

python tom_eval/tom_eval.py \
    --ckpt_path ./checkpoints/tom_sft_Qwen2.5-1.5B-Instruct/global_step_24 \
    --data_path ./tom_eval/tom_eval_datasets.csv \
    --max_model_len 2048 \
    --max_tokens 20

python tom_eval/tom_eval.py \
    --ckpt_path ./checkpoints/tom_sft_Qwen2.5-3B-Instruct/global_step_24 \
    --data_path ./tom_eval/tom_eval_datasets.csv \
    --max_model_len 2048 \
    --max_tokens 20

python tom_eval/tom_eval.py \
    --ckpt_path ./checkpoints/tom_sft_Qwen2.5-7B-Instruct/global_step_24 \
    --data_path ./tom_eval/tom_eval_datasets.csv \
    --max_model_len 2048 \
    --max_tokens 20

python tom_eval/tom_eval.py \
    --ckpt_path ./checkpoints/tom_sft_Qwen2.5-7B-Instruct-1M/global_step_24 \
    --data_path ./tom_eval/tom_eval_datasets.csv \
    --max_model_len 2048 \
    --max_tokens 20
