# -*- coding: utf-8 -*-
import requests
import ast
from tqdm import tqdm
import pandas as pd
import json
import os
import sys
from utils import *

url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = { 
    "Content-Type": "application/json", 
    "Authorization": "e33317e035514306ba0e31238dd3a2b74c5b9349ad5e441982e4194b4f42ab53"
}

def get_gpt_response(query):
    data = { 
        "model": "gpt-4", 
        "messages": [{"role": "user", "content": query}],  # Dynamically insert query here
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    answer = response.json()["choices"][0]["message"]["content"]
    # print("> Answer: ", answer)
    return answer

# Read previously saved index if available
retrieved_idx_path = 'retrieved_idx_set4_chain.json'
retrieved_chunks_path = 'retrieved_chunks_text_set4_chain.json'


with open('skylined_legal_chunks.json', 'r', encoding='utf-8') as f:
    legal_divided_chunks_raw = json.load(f)

with open('financial_docs_divided_chunks.json', 'r', encoding='utf-8') as f:
    financial_divided_chunks = json.load(f)

with open('skylined_paper_dict.json', 'r', encoding='utf-8') as f:
    paper_divided_chunks = json.load(f)

with open('skylined_legal_chunks2.json', 'r', encoding='utf-8') as f:
    legal_divided_chunks_raw2 = json.load(f)


    
if os.path.exists(retrieved_idx_path):
    with open(retrieved_idx_path, 'r', encoding='utf-8') as f:
        retrieved_chunks_idx = json.load(f)
    start_idx = len(retrieved_chunks_idx)
else:
    retrieved_chunks_idx = []
    start_idx = 0

if os.path.exists(retrieved_chunks_path):
    with open(retrieved_chunks_path, 'r', encoding='utf-8') as f:
        retrieved_chunks_query = json.load(f)
else:
    retrieved_chunks_query = []

data = pd.read_csv("parsed_set4_chain_gpt4omini.csv")
for idx in range(len(data)):
    data["doc"][idx] = ast.literal_eval(data["doc"][idx])

struct_info_dict = {}
file_list = get_all_files("doc/financial/")

try:
    for idx in tqdm(range(start_idx, len(data))):
        folder_name = data["type"][idx]
        folder_path = "doc/" + folder_name
        keep_doc_name = None
        
        chunks = {}
        doc_lst = data["doc"][idx]

        label = False
        multi_docs = False

        if folder_name == "paper":
            chunk_record = {}
            retrieve_chunks = {}
            for doc_name in doc_lst:
                retrieve_chunks[doc_name] = paper_divided_chunks[doc_name]["content"]
                chunk_record[doc_name] = [0]
            retrieved_chunks_idx.append(chunk_record)
            retrieved_chunks_query.append(retrieve_chunks)


        if folder_name == "legal":
            chunk_record = {}
            retrieve_chunks = {}
            parsed = data["parsed"][idx]
            if "判决结果" in parsed:
                legal_divided_chunks = legal_divided_chunks_raw2
            else:
                legal_divided_chunks = legal_divided_chunks_raw
            for doc_name in doc_lst:
                retrieve_chunks[doc_name] = legal_divided_chunks[doc_name]
                chunk_record[doc_name] = [0]
            retrieved_chunks_idx.append(chunk_record)
            retrieved_chunks_query.append(retrieve_chunks)
        
        if folder_name == "financial":
            input_str = data["parsed"][idx]
            type_word, merge_meta = parse_input(input_str)
            if type_word not in struct_info_dict:
                struct_info_dict[type_word] = {}
            struct_info_dict[type_word][idx] = merge_meta

            key = merge_meta[0][0]
            if key.endswith("Inc."):
                key = key[:-5]
            attribute = merge_meta[0][1]
            constraint = merge_meta[1]
            for doc_name in doc_lst:
                if key in doc_name:
                    label = True
                    break
            if label:
                doc_num = 0
                same_name_lst = []
                for name in financial_divided_chunks.keys():
                    if key in name:
                        doc_num += 1
                        same_name_lst.append(name)
                if doc_num > 1:
                    multi_docs = True
                    for doc_n in same_name_lst:
                        print(doc_n)
                        chunks[doc_n] = financial_divided_chunks[doc_n]
                else:
                    key = same_name_lst[0]
                    chunks = financial_divided_chunks[key]
            else:
                multi_docs = True
                for doc_name in doc_lst:
                    for name in financial_divided_chunks.keys():
                        if doc_name in name:
                            chunks[name] = financial_divided_chunks[name]
            
            instruction = "The given query is " + data["question"][idx] + " Please determine all possible chunks that may contain the answer to the given query. If you are not sure, you should include all possible ones. Please only return the index of the possible chunks like 9, 10, 12, 14. Do NOT give explanations."
            # print(data["question"][idx])
            chunks_cpy = chunks
            if multi_docs:
                chunk_record = {}
                retrieve_chunks = {}
                for doc_name in chunks_cpy.keys():
                    chunk_record[doc_name] = {}
                    retrieve_chunks[doc_name] = {}
                    retrieve_idx = []
                    chunks = chunks_cpy[doc_name]
                    for i, c_key in enumerate(chunks.keys()):
                        tmp = chunks[c_key]
                        if (attribute in tmp) or contains_word(tmp, attribute):
                            retrieve_idx.append(i)
                    str_keys = "{" + "; ".join([f"{i}: {c_key}" for i, c_key in enumerate(chunks.keys())]) + "}"
                    llm_query = str_keys + instruction
                    result = get_gpt_response(llm_query)
                    numbers = extract_numbers(result)
                    retrieve_idx.extend(num for num in numbers if num not in retrieve_idx)

                    key_list = list(chunks.keys())
                    tag = False
                    for num in numbers:
                        if num > len(chunks.keys()):
                            tag = True
                    if len(retrieve_idx) == 0 or tag:
                        retrieve_idx = list(range(len(chunks)))
                    print("retreived idx: ", retrieve_idx)
                    for num in retrieve_idx:
                        retrieve_chunks[doc_name][key_list[num]] = chunks[key_list[num]]
                    chunk_record[doc_name] = retrieve_idx
                retrieved_chunks_idx.append(chunk_record)
                retrieved_chunks_query.append(retrieve_chunks)
            else:
                retrieve_idx = []
                for i, c_key in enumerate(chunks.keys()):
                    tmp = chunks[c_key]
                    if (attribute in tmp) or contains_word(tmp, attribute):
                        retrieve_idx.append(i)
                str_keys = "{" + "; ".join([f"{i}: {c_key}" for i, c_key in enumerate(chunks.keys())]) + "}"
                
                llm_query = str_keys + instruction
                result = get_gpt_response(llm_query)
                numbers = extract_numbers(result)
                retrieve_idx.extend(num for num in numbers if num not in retrieve_idx)
                
                key_list = list(chunks.keys())
                
                if len(retrieve_idx) == 0:
                    retrieve_idx = list(range(len(chunks)))
                chunk_record = {key: retrieve_idx}
                retrieve_chunks = {key_list[num]: chunks[key_list[num]] for num in retrieve_idx}
                print("file name: ", key)
                print("retreived idx: ", retrieve_idx)
                
                retrieved_chunks_idx.append(chunk_record)
                retrieved_chunks_query.append(retrieve_chunks)

    with open(retrieved_idx_path, 'w', encoding='utf-8') as f:
        json.dump(retrieved_chunks_idx, f, ensure_ascii=False, indent=4)
    with open(retrieved_chunks_path, 'w', encoding='utf-8') as f:
        json.dump(retrieved_chunks_query, f, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"Error encountered: {e}. Exiting...")
    with open(retrieved_idx_path, 'w', encoding='utf-8') as f:
        json.dump(retrieved_chunks_idx, f, ensure_ascii=False, indent=4)
    with open(retrieved_chunks_path, 'w', encoding='utf-8') as f:
        json.dump(retrieved_chunks_query, f, ensure_ascii=False, indent=4)
    sys.exit()
