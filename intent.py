import openai
import math
import json
import asyncio
from typing import Any, List
from openai_generate import OpenAIChat

openai.api_key = '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'
chat = OpenAIChat()

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
        with open(path, "a+", encoding='utf-8') as f:
            for _data in data:
                print(_data)
                f.write(json.dumps(_data, ensure_ascii=False))
                f.write(',\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for _data in data:
                f.write(_data)
                f.write('\n')
    f.close()

def get_text_questionnaire(generated_json):
    generated_list = []
    for questionnaire in generated_json:
        _questionnaire = ''
        _questionnaires = questionnaire['qa']
        for _temp in _questionnaires:
            _questionnaire += _temp['question']
            if 'options' in _temp and len(_temp['options']) > 1:
                _questionnaire += ' '.join(_temp['options'])
        generated_list.append({"title": questionnaire['title'], "content": _questionnaire})
    
    return generated_list
    

async def run_with_batch_generate(dataset, save_path, rerun=False, rerun_indices=[]):
    sample_list = dataset
    rerun_elements = sample_list if not rerun else [sample_list[i] for i in rerun_indices]
    batch_size = 4
    num_batches = math.ceil(len(rerun_elements) / batch_size) # 5
        
    for i in range(num_batches):
        print(i)
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(rerun_elements))

        messages_list = [[{"role": "user", "content": example['prompt'] if 'prompt' in example else ''}] for example in rerun_elements[batch_start:batch_end]]
        # print(messages_list)
        responses =  await chat.async_run(messages_list, List)

        for j, response in enumerate(responses):
            index = batch_start + j if rerun == False else rerun_indices[batch_start + j]
            if response is None:
                sample_list[index].update({
                        'target': 'None'})
            else:
                sample_list[index].update({
                        'target': response,
                    })
        
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in sample_list:
                    json_str = json.dumps(item, ensure_ascii=False)
                    f.write(json_str + '\n')

questionnaires_dataset = read_data('example.jsonl')
batch_results = asyncio.run(run_with_batch_generate(questionnaires_dataset, 'all_datast_with_target.jsonl'))
