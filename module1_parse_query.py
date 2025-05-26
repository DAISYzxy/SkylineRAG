# -*- coding: utf-8 -*-
import requests
import json
import re
import pandas as pd
from tqdm import tqdm


def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Try loading the JSON object from each line
                data.append(pd.json.loads(line.strip()))  # strip to remove any leading/trailing whitespace
            except ValueError as e:
                print(f"Error parsing line: {e}")
                continue  # skip malformed lines
    return pd.DataFrame(data)



url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = { 
    "Content-Type": "application/json", 
    "Authorization": "e33317e035514306ba0e31238dd3a2b74c5b9349ad5e441982e4194b4f42ab53"
}

def get_gpt_response(query):
    data = { 
        "model": "gpt-3.5-turbo", 
        "messages": [{"role": "user", "content": query}],  # Dynamically insert query here
        "temperature": 0
    }
    
    # response = requests.post(url, headers=headers, data=json.dumps(data))
    # response_content = response.json()["choices"][0]["message"]["content"]
    # cleaned_string = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    answer = response.json()["choices"][0]["message"]["content"]
    print("> Answer: ", answer)
    return answer

# Example usage:


file_path = 'loong.jsonl'
df = jsonl_to_dataframe(file_path)
set_1 = df[df['set'] == 4]
spotlight = set_1[set_1["level"] == 4].reset_index().drop(columns="index")
instruction = "Given the question and instruction as a query, please parse the query into meta queries. The meta query consists of (1) extraction: choose from the following four forms - (key, attribute), (head, relationship, tail), (timestamp, value), pure text which is the information you need to extract from the external documents; (2) Data Structure: choose from [table, graph, time series, pure text]; (3) Further implementations: ranking/graph chain construction/conditional selection/... ((3) can be void for some queries. ) For example 1, (1) extract: (公司, 利润总额); (2) Data Structure: Table; (3) conditional selection: 高利润(1,000,000,000.00以上)，中利润 (100,000,000.00以上且1,000,000,000.00以下)，低利润(0以上且100,000,000.00以下)，负利润(0及0以下). For example 2, (1) extract: (paper name, cite, paper name); (2) Data Structure: Graph; (3) construct a citation chain. For example 3, (1) extract: (判决书, 案由/判决结果); (2) Data Structure: Table; (3) 按照案由/判决结果分类. Please strictly follows the previous formats. Give NO explanations."
spotlight["parsed"] = None
for idx in tqdm(range(len(spotlight))):
    type = spotlight["type"][idx]
    if type == "paper":
        query_text = spotlight["instruction"][idx] + "\n" + instruction
    else:
        print(spotlight["question"][idx])
        query_text = spotlight["question"][idx] +"\n" + instruction
    result = get_gpt_response(query_text)
    spotlight["parsed"][idx] = result

spotlight.to_csv("parsed_set4_chain_gpt4omini.csv", index=False)

