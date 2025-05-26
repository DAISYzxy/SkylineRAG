import json
import pandas as pd
import ast
from collections import defaultdict
import heapq
from tqdm import tqdm
from utils import *
from openai import AzureOpenAI, OpenAI
import os
import tiktoken
from transformers import AutoTokenizer, LlamaTokenizer
from functools import lru_cache
import sys




# GPT 系列 (使用 tiktoken 的 cl100k_base 编码)
@lru_cache(maxsize=None)
def get_gpt_encoder():
    return tiktoken.get_encoding("cl100k_base")

def count_gpt_tokens(text: str) -> int:
    encoder = get_gpt_encoder()
    return len(encoder.encode(text))



def get_qwen_response(
    user_prompt:str = None,
    num_responses:int = 1,
    model_name:str = "Qwen/Qwen2.5-72B-Instruct",
    base_url:str = "",
    api_key:str = ""
):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        n=num_responses
    )
    responses = [choice.message.content for choice in response.choices]

    if len(responses) == 1:
        return responses[0]
    else:
        return responses

def greedy(data):
    # 1. 构建 chunk 到 query 的映射
    chunk_to_query = defaultdict(set)
    for query, chunks in data.items():
        for chunk in chunks:
            chunk_to_query[chunk].add(query)

    # 2. 初始化候选 chunks 集合
    all_queries = set(data.keys())  # 需要覆盖的 query 集合
    covered_queries = set()
    selected_chunks = []
    selected_chunk_queries = {}  # 存储每个选中的 chunk 及其覆盖的 queries

    # 3. 使用贪心算法选择 chunk
    while covered_queries != all_queries:
        best_chunk = None
        best_gain = -1
        
        for chunk, queries in chunk_to_query.items():
            uncovered = queries - covered_queries
            if not uncovered:
                continue
            gain = len(uncovered) / len(queries)  # G = |Q(C) ∩ (Q - M)| / Token(C)
            if gain > best_gain:
                best_gain = gain
                best_chunk = chunk
        
        if best_chunk is None:
            break  # 无法继续覆盖 query
        
        # 选中最佳 chunk
        selected_chunks.append(best_chunk)
        selected_chunk_queries[best_chunk] = list(chunk_to_query[best_chunk])
        covered_queries.update(chunk_to_query[best_chunk])
        del chunk_to_query[best_chunk]

    # 输出结果
    print("Selected chunks:", selected_chunks)
    print("Chunk to Query Mapping:")
    for chunk, queries in selected_chunk_queries.items():
        print(f"Chunk {chunk}: Queries {queries}")
    return selected_chunks, selected_chunk_queries, chunk_to_query



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

def process_data(data):
    new_dict = {}
    accum = 0
    
    for item in data:
        for key in item.keys():
            if key not in new_dict:
                accum += 1
                new_dict[key] = accum * 100
    
    for item in data:
        for key, values in item.items():
            base_value = new_dict[key]
            item[key] = [base_value + i for i in values]
    
    return new_dict, data


def split_long_chunk(text, instruction, question_suffix, max_tokens=28000):
    """如果文本超过指定长度，按句号拆分成多个小块。"""
    reserved_tokens = count_gpt_tokens(instruction + question_suffix)
    max_chunk_length = max_tokens - reserved_tokens
    
    sentences = text.split('。')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if count_gpt_tokens(current_chunk + sentence) > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"
        else:
            current_chunk += sentence + "。"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


data_set1 = pd.read_csv("parsed_set1_comparison_gpt4o.csv")
data_set2 = pd.read_csv("parsed_set2_comparison_gpt4o.csv")
data_set3 = pd.read_csv("parsed_set3_comparison_gpt4o.csv")
data_set4 = pd.read_csv("parsed_set4_comparison_gpt4o.csv")

spotlight = pd.concat([data_set1, data_set2, data_set3, data_set4], ignore_index=True)
for idx in range(len(spotlight)):
    spotlight["doc"][idx] = ast.literal_eval(spotlight["doc"][idx])

with open('retrieved_idx_set1_comparison.txt', 'r', encoding='utf-8') as f:
    retrieved_idx_set1 = json.load(f)

with open('retrieved_idx_set2_comparison.txt', 'r', encoding='utf-8') as f:
    retrieved_idx_set2 = json.load(f)

with open('retrieved_idx_set3_comparison.txt', 'r', encoding='utf-8') as f:
    retrieved_idx_set3 = json.load(f)

with open('retrieved_idx_set4_comparison.txt', 'r', encoding='utf-8') as f:
    retrieved_idx_set4 = json.load(f)


retrieved_idx = []

for item in retrieved_idx_set1:
    retrieved_idx.append(item)
for item in retrieved_idx_set2:
    retrieved_idx.append(item)
for item in retrieved_idx_set3:
    retrieved_idx.append(item)
for item in retrieved_idx_set4:
    retrieved_idx.append(item)


data = retrieved_idx

new_dict, data = process_data(data)

updated_data = {}
inverted_chunk_name = {}
for idx in range(len(data)):
    tmp_dict = data[idx]
    updated_data[idx] = []
    for key in tmp_dict.keys():
        tmp_chunk = tmp_dict[key]
        for item in tmp_chunk:
            inverted_chunk_name[item] = key
            updated_data[idx].append(item)


selected_chunks, selected_chunk_queries, rest_chunks = greedy(updated_data)
chunk_to_query = {}
for query, chunks in updated_data.items():
    for chunk in chunks:
        if chunk not in chunk_to_query:
            chunk_to_query[chunk] = []
        chunk_to_query[chunk].append(query)


struct_info_dict = {}
new_spotlight = {}
new_spotlight["key"] = []
new_spotlight["attribute"] = []
new_spotlight["constraint"] = []
new_spotlight["chunk_list"] = []
new_spotlight["type"] = []
new_spotlight["set"] = []
new_spotlight["question"] = []
new_spotlight["answer"] = []
new_spotlight["doc"] = []

for idx in tqdm(range(len(spotlight))):
    if spotlight["type"][idx] == "paper":
        new_spotlight["key"].append(spotlight["question"][idx])
        new_spotlight["attribute"].append("cite/reference")
        new_spotlight["constraint"].append("graph")
        new_spotlight["chunk_list"].append(updated_data[idx])
        new_spotlight["set"].append(spotlight["set"][idx])
        new_spotlight["type"].append(spotlight["type"][idx])
        new_spotlight["question"].append(spotlight["instruction"][idx])
        new_spotlight["answer"].append(spotlight["answer"][idx])
        new_spotlight["doc"].append(spotlight["doc"][idx])

    if spotlight["type"][idx] != "paper":
        parsed = spotlight["parsed"][idx]
        type_word, merge_meta = parse_input(parsed)
        if type_word not in struct_info_dict:
            struct_info_dict[type_word] = {}
        struct_info_dict[type_word][idx] = merge_meta

        key = merge_meta[0][0]
        if key.endswith("Inc."):
            key = key[:-5]
        if key.endswith("标题"):
            key = key[:-2]
        attribute = merge_meta[0][1]
        constraint = merge_meta[1]
        if constraint and "conditional selection: " in constraint:
            constraint = constraint[23:]
        new_spotlight["key"].append(key)
        new_spotlight["attribute"].append(attribute)
        new_spotlight["constraint"].append(constraint)
        new_spotlight["chunk_list"].append(updated_data[idx])
        new_spotlight["set"].append(spotlight["set"][idx])
        new_spotlight["type"].append(spotlight["type"][idx])
        new_spotlight["question"].append(spotlight["question"][idx])
        new_spotlight["answer"].append(spotlight["answer"][idx])
        new_spotlight["doc"].append(spotlight["doc"][idx])

df_new_spotlight = pd.DataFrame(new_spotlight)


with open('skylined_legal_chunks.json', 'r', encoding='utf-8') as f:
    legal_divided_chunks_raw = json.load(f)

with open('skylined_legal_chunks2.json', 'r', encoding='utf-8') as f:
    legal_divided_chunks_raw2 = json.load(f)

with open('financial_docs_divided_chunks.json', 'r', encoding='utf-8') as f:
    financial_divided_chunks = json.load(f)

with open('skylined_paper_dict.json', 'r', encoding='utf-8') as f:
    paper_divided_chunks = json.load(f)



instruction = "\n\n Please answer the question based on the above text without giving any explanations. If there is no corresponding answer, just return 'void'."
answer_4_selected_chunk = {}



legal_divided_chunks = legal_divided_chunks_raw

# 从511开始处理
for chunk_key in tqdm(list(chunk_to_query.keys())[0:]):
    # print(f"🔄 正在处理 chunk_key: {chunk_key}")
    query_lst = chunk_to_query[chunk_key]  # the query list for new_spotlight
    doc_name = inverted_chunk_name[chunk_key]
    chunk_idx_in_doc = chunk_key - new_dict[doc_name]

    for query_idx in query_lst:
        question = df_new_spotlight["question"][query_idx]
        key = df_new_spotlight["key"][query_idx]
        attribute = df_new_spotlight["attribute"][query_idx]
        query_type = df_new_spotlight["type"][query_idx]

        if query_type == "legal":
            if attribute == "判决结果":
                legal_divided_chunks = legal_divided_chunks_raw2
                chunk_text = legal_divided_chunks[doc_name]
                question_suffix = "以上判决书的判决结果是什么？"
            else:
                legal_divided_chunks = legal_divided_chunks_raw
                chunk_text = legal_divided_chunks[doc_name]
                question_suffix = "以上判决书的案由是什么？"
            
            # 如果超过最大长度，进行切分
            sub_chunks = split_long_chunk(chunk_text, instruction, question_suffix, max_tokens=28000)
            collected_answers = []

            # 遍历每个小段，逐个询问
            for sub_chunk in sub_chunks:
                query = sub_chunk + instruction + question_suffix
                total_tokens = count_gpt_tokens(query)
                
                # ⚠️ 二次校验长度是否超出
                if total_tokens > 30000:
                    print(f"❌ 超出最大长度限制，跳过。Tokens: {total_tokens}")
                    continue
                
                # print(f"🔍 正在查询: {query[:50]}... (Tokens: {total_tokens})")
                answer = get_qwen_response(query)

                if answer != "void":
                    collected_answers.append(answer)

            # 存储非空答案
            if collected_answers:
                if doc_name not in answer_4_selected_chunk:
                    answer_4_selected_chunk[doc_name] = {}
                answer_4_selected_chunk[doc_name][attribute] = " ".join(collected_answers)

        if query_type == "financial":
            doc_text = financial_divided_chunks[doc_name]
            chunk_text = doc_text[list(doc_text.keys())[chunk_idx_in_doc]]
            question_suffix = f"({key}, {attribute})"
            
            # 如果超过最大长度，进行切分
            sub_chunks = split_long_chunk(chunk_text, instruction, question_suffix, max_tokens=28000)
            collected_answers = []

            for sub_chunk in sub_chunks:
                query = sub_chunk + instruction + question_suffix
                total_tokens = count_gpt_tokens(query)
                
                # ⚠️ 二次校验长度是否超出
                if total_tokens > 30000:
                    print(f"❌ 超出最大长度限制，跳过。Tokens: {total_tokens}")
                    continue

                # print(f"🔍 正在查询: {query[:50]}... (Tokens: {total_tokens})")
                answer = get_qwen_response(query)

                if answer != "void":
                    collected_answers.append(answer)

            # 存储非空答案
            if collected_answers:
                if doc_name not in answer_4_selected_chunk:
                    answer_4_selected_chunk[doc_name] = {}

                if chunk_idx_in_doc not in answer_4_selected_chunk[doc_name]:
                    answer_4_selected_chunk[doc_name][chunk_idx_in_doc] = {}

                answer_4_selected_chunk[doc_name][chunk_idx_in_doc][attribute] = " ".join(collected_answers)

    # 每处理完一个 chunk_key 就保存进度，防止丢失
    with open('chunks_answer_comparison_qwen.json', 'w', encoding='utf-8') as f:
        json.dump(answer_4_selected_chunk, f, ensure_ascii=False, indent=4)
    print(f"✅ 已保存进度: {chunk_key}")
