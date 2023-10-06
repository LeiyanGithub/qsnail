# add msg
import json
import argparse
import json
import math
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args
import re


import json
import re
from json import JSONDecodeError
from typing import Any, List

from langchain.schema import BaseOutputParser, OutputParserException

# import re

def is_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(text)
    return match is not None

def read_data(path):
    if 'json' in path:
        with open(path, "r") as f:
            datas = json.load(f)
    else:
        with open(path, "r") as f:
            datas = [_data.strip('\n') for _data in f.readlines()]
    
    print(len(datas))

    return datas


def write_file(path, data):
    if 'json' in path:
        with open(path, "w", encoding="utf-8") as f:
            for _data in data:
                # print(_data)
                f.write(json.dumps(_data, ensure_ascii=False))
                f.write(',\n')
    else:

        with open(path, 'a+') as f:
            for _data in data:
                f.write(_data)
                f.write('\n')
    f.close()

def get_prompt(model_name, users, assistants, is_save=False):
    # model_name = 'gpt-3.5-turbo'
    # 根据user和assistant内容生成promt
    conv = get_conversation_template(model_name)

    # print(len(users), len(assistants))
    
    for idx in range(len(assistants)):
        conv.append_message(conv.roles[0], users[idx])
        conv.append_message(conv.roles[1], assistants[idx])
    
    for idx in range(len(users) - len(assistants)):
        conv.append_message(conv.roles[0], users[len(assistants) + idx])
        conv.append_message(conv.roles[1], None)
    
    if model_name in ['gpt-3.5-turbo', 'gpt-4']:
        prompt = conv.to_openai_api_messages()
    else:
        if is_save:
            prompt = conv.to_openai_api_messages()
        else:
            prompt = conv.get_prompt()
    return prompt


def generate(model, tokenizer, temperature=0.7, repetition_penalty=1.0, max_new_tokens=1024, messages=[], device = "cuda"):
    request_prompt = [value for _, value in messages.items()]
    # print("request_prompt: ", request_prompt)
    with torch.no_grad():
        inputs = tokenizer(request_prompt, return_token_type_ids=False, padding=True, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        del inputs

        results_outputs = {}
        outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )

        del output_ids
    # print(outputs)
    index = 0 
    for key, value in messages.items():
        # results_outputs[key] = outputs[index].split("\n\n答：")[1]
        # print("outputs before: ", outputs[index])
        results_outputs[key] = outputs[index].split("ASSISTANT:")[1]
        # print("outputs: ", results_outputs[key])
        index += 1
    
    return results_outputs

def batch_generate(temperature=0.7, repetition_penalty=1.0, max_new_tokens=2048, messages={}, device="cuda"):
    global model
    global tokenizer
    global results_outputs

    print("新生成长度：", len(messages))
    start = time.time()
    output = generate(model, tokenizer, temperature=0.7, repetition_penalty=1.0, max_new_tokens=2048, messages=messages, device="cuda")
    end = time.time()
    run_time = (end - start) / 60
    print("一个{}需要运行{}".format(len(messages), run_time))
    
    regenerate = {}
    for key, value in output.items():
        if len(value) < 10 or not is_chinese(value) :
            regenerate[key] = messages[key]
        else:
            results_outputs[key] = value
    
    return results_outputs, regenerate


def generate_questionnaire(temperature=0.7, repetition_penalty=1.0, max_new_tokens=1024, messages=[], save_path='./results_v1/', batch_size=8):
    global results_outputs
    
    new_regenerate = {}
    num_batches = math.ceil(len(messages) / batch_size)
    print("总batch: ", num_batches)
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(messages))
        keys = list(messages.keys())[batch_start: batch_end]
        new_messages = {key: messages[key] for key in keys}

        # 返回需要再次生成的case
        output, need_regenerate = batch_generate(temperature=0.7, repetition_penalty=1.0, max_new_tokens=2048, messages=new_messages, device="cuda")
        
        for key, value in need_regenerate.items():
            new_regenerate[key] = messages[key]
        
        results = []
        for key, value in results_outputs.items():
            results.append({key: value})
        write_file(save_path, results)
        
    return new_regenerate


def get_message(model_name, data_path, begin_index, split_size):
    questionnaires = read_data(data_path)[begin_index: begin_index + split_size]

    messages = {}
    raw_message = {}

    for _questionnaire in questionnaires:
        prompt = "作为问卷设计专家，请以" + _questionnaire['title'].replace("问卷", "") + "为主题设计问卷。" + _questionnaire['target'] + """根据上述要求生成问卷的标题，序列化的问题以及选项。注意：
1.	每个问题需要主题有关，不必要的问题不应该出现，一个问题不能出现多个小问题。总问题的数量不能过少也不能过多，在20个以上30个以内都是合理的。
2.	每个问题的尽量包含对应合理且详细的选项，尽量不要生成开放性的题目。
3.	问题之间的顺序要求是有逻辑的。
4.  选项应该尽量详细
"""
        messages[_questionnaire['title']] = get_prompt(model_name, users=[prompt], assistants=[])
        raw_message[_questionnaire['title']] = prompt
    
    return messages, raw_message


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--model_name", type=str, default="vicuna-7b")
    parser.add_argument("--model_path", type=str, default="/home/liulian/yan/pytorch_model/vicuna-7b-v1.5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prompt", type=str, default="only_topic")
    parser.add_argument("--loc", type=str, default="only_topic")
    parser.add_argument("--begin_index", type=int, default=0)
    parser.add_argument("--split_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="../dataset/test_target.jsonl")
    parser.add_argument("--save_path", type=str, default="../results/")
    
    args = parser.parse_args()

    device = "cuda"
    model, tokenizer = load_model(
        args.model_path,
        device=device,
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=False,
        revision='main',
        debug=False,
    )

    model.eval()
    
    prompt, _ = get_message(args.model_name, args.data_path, args.begin_index, args.split_size)
    # print(len(prompt))
    save_path = args.save_path + args.prompt + '/' + args.model_name + '_' + args.loc + '.jsonl'

    # regenerate = {}
    # regenerate = {_data.keys()[0]: _data.values()[0] for _data in read_data(save_path)}
    # if not regenerate:
    regenerate = prompt
    
    # 记录一个最终版的result_output, 每次只需要更新key对应的内容即可
    results_outputs = {}
    for key, value in prompt.items():
        results_outputs[key] = ""

    while regenerate:
        # 只要有需要regenerate就需要继续生成
        regenerate = generate_questionnaire(temperature=0.7, repetition_penalty=1.0, max_new_tokens=1024, messages=regenerate, save_path=save_path, batch_size=args.batch_size)
