### verl install

```
conda create -n multiturn python==3.10
conda activate multiturn

pip3 install torch torchvision
# pip3 install ninja, packaging
pip3 install flash-attn --no-build-isolation
git clone git@github.com:chunhuizhang/multi-turn-verl.git
cd multi-turn-verl
# conda install nvidia::cuda-toolkit
# for torch-memory-saver, maybe 需要在计算节点安装（libcuda.so）
# for zsh
pip3 install -e ".[sglang]"

# misc
# for ray debug
pip install debugpy==1.8.0
```


### gsm8k sglang multi-turn

```
cd examples/data_preprocess
python3 gsm8k.py

cd multi-turn-verl
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

