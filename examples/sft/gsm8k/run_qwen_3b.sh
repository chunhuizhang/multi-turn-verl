# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# python ./examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/gsm8k/train.parquet \
    data.val_files=./data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-3b \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@