# -*- coding: utf-8 -*-
import os
import json
import re
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

def extract_numbers(text):
    return [int(num) for num in re.findall(r'\d+', text)]


def get_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def read_file(type, folder_path, given_docs):
    if type == "legal":
        folder_path += "/legal.json"
        with open(folder_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        docs = {}
        for doc in given_docs:
            for key in content.keys():
                if doc in key:
                    docs[doc] = content[key]
        return docs
    if type == "financial":
        file_list = get_all_files(folder_path)
        docs = {}
        for doc in given_docs: # given_docs = set_1_spotlight["doc"][test_idx]
            for file in file_list:
                if doc in file:
                    file_content = read_txt_file(file)
                    docs[doc] = file_content
        return docs

def parse_input(input_str):

    # 修改正则表达式，支持 extraction 和 extract
    extract_meta_match = re.search(r'\(1\)\s*(extraction|extract):\s*\(([^)]+)\)', input_str)
    if not extract_meta_match:
        raise ValueError("Could not extract (1) section.")
    
    # 提取所有逗号分隔的部分
    items = extract_meta_match.group(2).split(', ')
    
    # 去除每个项的空格
    items = [item.strip() for item in items]
    if len(items) == 1:
        raise ValueError("Extraction Meta Query Error!")
    
    if len(items) == 2:
        key, attribute = items
        constraint = None
    elif len(items) == 3:
        key, attribute, constraint = items
    
    # 提取（2）中冒号后面的词（Data Structure: Table）
    data_structure_match = re.search(r'\(2\)\s*Data Structure:\s*([^\;]+)', input_str)
    if not data_structure_match:
        raise ValueError("Could not extract (2) section.")
    
    type_word = data_structure_match.group(1).strip()

    # 清理type中的特殊字符，去除换行符和回车符
    type_word = re.split(r'[;\n]+', type_word)[0]
    type_word = type_word.strip()  # 进一步去除两端的空格

    # 提取（3）部分，只提取 (3) 后的内容
    imple_meta = None
    condition_match = re.search(r'\(3\)\s*(.*)', input_str)
    if condition_match:
        imple_meta = condition_match.group(1).strip()
        # 如果是 "void"，则设置为 None
        if imple_meta.lower() == "void":
            imple_meta = None
    
    if type_word.lower() == "table":
        type_word = "Table"  # 保持类型为"Table"

    if (constraint != None) and (constraint not in imple_meta):
        merge_meta = ((key, attribute), constraint)
    else:
        merge_meta = ((key, attribute), imple_meta)

    return type_word, merge_meta


def contains_word(text, word): # re.IGNORECASE: 匹配不区分大小写
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))


def split_text_into_chunks(text):
    # 使用正则表达式匹配可能的页码，确保是居中的数字
    pages = re.split(r'\n\s{3,}(\d+)\s{3,}\n', text)  # 至少3个空格，模拟居中对齐
    
    chunks = []
    for i in range(1, len(pages), 2):
        page_number = pages[i].strip()
        page_content = pages[i + 1].strip()
        chunks.append(page_content)
    headings = None
    if len(chunks) == 0:
        pattern = r"[一二三四五六七八九十]+、[^（\n]+"
        headings = re.findall(pattern, text)
        last_pos = 0
        
        for heading in headings:
            # Find the position of each heading in the text
            start_pos = text.find(heading, last_pos)
            next_pos = text.find(heading, start_pos + 1) if text.find(heading, start_pos + 1) != -1 else len(text)
            
            # Extract chunk from last position to the current heading
            chunk = text[last_pos:start_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            last_pos = start_pos
        
        # Append the last chunk after the final heading
        final_chunk = text[last_pos:].strip()
        if final_chunk:
            chunks.append(final_chunk)
    return chunks, headings


def get_sentence_embedding(sentence):
    # Tokenize句子并将其转换为tensor
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取[CLS] token对应的embedding作为句子的表示
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return sentence_embedding