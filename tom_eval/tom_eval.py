import argparse
import os
import pandas as pd
import torch
from vllm import LLM, SamplingParams  
from datasets import load_dataset
import re
from transformers import AutoTokenizer


def generate(llm, prompts, use_tqdm=False, args=None):    
    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    n=args.n)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    responses = [[output.text for output in output_item.outputs] for output_item in outputs]
    return responses

def compute_score(resp, answer):
    try:
        resp = resp.lower().strip()
        answer = answer.lower().strip()
        return int(resp.startswith(answer))
    except:
        print(f"Error: {resp} {answer}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/tom_sft_Qwen2.5-3B-Instruct/global_step_24")
    parser.add_argument("--data_path", type=str, default="./tom_eval/tom_eval_datasets.csv")
    # save_dir
    parser.add_argument("--save_dir", type=str, default="./tom_eval/results")
    parser.add_argument("--max_model_len", type=int, default=1024*2)
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model_id = re.sub(r'\.', '-', re.search(r'tom_sft_(.*?)/global_step', args.ckpt_path).group(1))
    step = int(re.search(r'global_step_(\d+)', args.ckpt_path).group(1))

    eval_ds = load_dataset("csv", data_files=args.data_path)["train"]

    llm = LLM(
        model=args.ckpt_path,
        # model='/data/hf/zch/checkpoints',
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    # tokenizer = AutoTokenizer.from_pretrained('/data/hf/zch/checkpoints')

    # llm.llm_engine.tokenizer.eos_token_id = 151643

    prompts = []
    for example in eval_ds:
        question = f"Story: {example['story']}\nQuestion: {example['question']}"
        message = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    responses = generate(llm, prompts, use_tqdm=True, args=args)

    total_score = 0
    results = []
    for resp, example in zip(responses, eval_ds):
        score = compute_score(resp[0], example["answer"])
        total_score += score
        example['response'] = resp[0]
        example["score"] = score
        results.append(example)
        print(f'score: {score} resp: {resp[0]} answer: {example["answer"]}')
        print('-'*100)

    print(f"Total score: {total_score / len(responses)}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, f"{model_id}_step_{step}_results.csv"), 
                      index=False, 
                      encoding="utf-8-sig")
