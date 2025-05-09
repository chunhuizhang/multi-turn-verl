#!/bin/bash
set -x

MODEL_LIST=("Qwen/Qwen2.5-0.5B-Instruct" 
            "Qwen/Qwen2.5-1.5B-Instruct"
            "Qwen/Qwen2.5-3B-Instruct"
            "Qwen/Qwen2.5-7B-Instruct"
            "Qwen/Qwen2.5-7B-Instruct-1M")

for model in ${MODEL_LIST[@]}; do
    echo "Training $model"
    model_base_name=$(basename $model)
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/tom_sft/train.parquet \
        data.val_files=./data/tom_sft/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['question'] \
        +data.response_dict_keys=['answer'] \
        data.micro_batch_size_per_gpu=4 \
        model.partial_pretrain=$model \
        model.enable_gradient_checkpointing=True \
        trainer.default_local_dir=./checkpoints/tom_sft_${model_base_name} \
        trainer.project_name=tom-sft \
        trainer.experiment_name=tom-sft-${model_base_name} \
        trainer.total_epochs=5 \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null $@
done