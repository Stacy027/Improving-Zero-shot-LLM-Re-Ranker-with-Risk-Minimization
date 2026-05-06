import os
import argparse
import json
import re
import string

import torch
from tqdm import tqdm

from utils.file_utils import print_args
from utils.model_utils import load_model_and_tokenizer
from collections import Counter

MAX_LENGTH = 2048
def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example, num_docs=1):
    if num_docs == 0:
        question_text = normalize_question(example["question"])
        ex_prompt = f"Answer these questions:\nQ: {question_text}\nA:"
    elif num_docs == 1:
        q = normalize_question(example["question"])
        title = example['ctxs'][0]['title']
        text = example['ctxs'][0]['text']
        ex_prompt = f"{title}\n\n{text}\n\nBased on this text, answer the question:\nQ: {q}\nA:"
    else:
        docs_text = []
        for i in range(num_docs):
            # if i == 0:
            docs_text.append(f"{example['ctxs'][i]['title']}\n\n{example['ctxs'][i]['text']}")
            # else:
                # if example["ctxs"][i]["has_answer"]:
                    # docs_text.append(f"{example['ctxs'][i]['title']}\n\n{example['ctxs'][i]['text']}")
        q = normalize_question(example["question"])
        docs_text = "\n\n".join(docs_text)
        # docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:num_docs]])
        ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
        # num_docs = len(docs_text)
    return ex_prompt#, num_docs


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_answer_from_model_output(outputs, tokenizer, prompt):
    generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    generation_str = generation_str[len(prompt):]
    answer = generation_str.split("\n")[0]
    return answer, generation_str


def evaluate_dataset(
        model, tokenizer, device, eval_dataset, max_length, num_docs=0, output_dir=None, max_tokens_to_generate=10
):
    idx = 0
    num_correct = 0
    num_has_answer = 0
    num_too_long = 0
    sample_prompt = None
    f1_total = 0
    # eval_dataset = eval_dataset[:200]
    for ex in (tq := tqdm(eval_dataset, desc=f"EM:  0.0%")):
        answers = ex["answers"]
        prompt = build_qa_prompt(ex, num_docs=num_docs)
        if idx == 0:
            sample_prompt = prompt
        has_answer = text_has_answer(answers, prompt)
        max_length = max(len(tokenizer.encode(answer, add_special_tokens=False)) for answer in answers)
        if max_length:
            max_tokens_to_generate = max_length
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        if input_ids.shape[-1] > MAX_LENGTH - max_tokens_to_generate:
            num_too_long += 1
            input_ids = input_ids[..., -(MAX_LENGTH - max_tokens_to_generate):]
            attention_mask = attention_mask[..., -(MAX_LENGTH - max_tokens_to_generate):]
        # attention_mask = torch.ones_like(input_ids) 
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens_to_generate)                
                # input_ids, max_new_tokens=max_tokens_to_generate)
        # generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        # prediction = generation_str[len(prompt):]
        # prediction = prediction.split("\n")[0]
        prediction, generation = get_answer_from_model_output(outputs, tokenizer, prompt)
        is_correct = any([exact_match(prediction, answer) for answer in answers])
        f1 = max([f1_score(prediction, answer) for answer in answers])
        f1_total += f1
        idx += 1
        if is_correct:
            num_correct += 1
        if has_answer:
            num_has_answer += 1
        tq.set_description(f"EM: {num_correct / idx * 100:4.1f}%")

    em = num_correct / idx * 100
    f1_score_mean = f1_total / idx * 100
    has_answer = num_has_answer / idx * 100
    print(f"EM: {em:.2f}%")
    print(f"F1: {f1_score_mean:.2f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    if output_dir is not None:
        d = {"em": em, "f1": f1_score_mean, "has_answer": has_answer, "num_examples": idx, "too_long": num_too_long}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def main(args):
    dataset_name = re.search(r"([^/]+)\.json", args.dataset_path).group(1)
    model_name = re.search(r"([^/]+)$", args.model_name).group(1)
    output_dir = f'{args.output_file}/{dataset_name}-doc{args.num_docs}-{model_name}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print_args(args, output_dir=output_dir)

    print("Loading model:", args.model_name)
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    tokenizer.pad_token_id = 0

    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings

    eval_dataset = load_dataset(args.dataset_path)

    evaluate_dataset(
        model, tokenizer, device, eval_dataset,
        max_length=model_max_length,
        num_docs=args.num_docs,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument("--model_name", type=str, default="./llama-2-7b-hf")
    parser.add_argument("--model_parallelism", action="store_true", default=True)
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_docs", type=int, default=5)
    
    # Dataset params
    parser.add_argument("--dataset_path", type=str)#, default='./datasets/')
    parser.add_argument("--output_file", type=str, default='./datasets/')
    args = parser.parse_args()
    
    main(args)
