import os
import gc
import time
import warnings

import pandas as pd
import polars as pl
import uuid
import json
from typing import List
from datetime import datetime
import psycopg2
import re
import random
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Login using e.g. `huggingface-cli login` to access this dataset

start_tag = '<keep>'
end_tag = '</keep>'

test_trace = "Hello, I'm Samuel and it's a great day today!"

def return_para_w_last_sentence_wrapped(para: str, last_n_sentences: int = 2):
    # Match a period, space, and then a capital letter
    eos_regex = r'[.!?]\s[A-Z]'
    offset = 2
    
    positions = [m.start() for m in re.finditer(eos_regex, para)]
    if positions == []:
        positions = [0]
        offset = 0
    if len(positions) < last_n_sentences:
        last_n_sentences = len(positions)
    last_sentence_start = positions[-1*last_n_sentences]+offset
    last_sentence = para[last_sentence_start:]
    wrapped_sentence = f'{start_tag}{last_sentence}{end_tag}'
    wrapped_paragraph = f'{para[:last_sentence_start]}{wrapped_sentence}\n\n'
    return wrapped_paragraph

def insert_keep_text_at_eop(text, config: dict = {}):
    default_config = {
        'k': 3,
        'min_p_length': 300
    }
    if len(config) == 0:
        config = default_config
    else:
        config = {**default_config, **config}
    # Wrap the last sentence of every paragraph with length larger than min_p_length with <keep> tags.
    paragraphs = text.split('\n\n')
    text_wrapped = ''
    ct_para = 0
    ct_sen = 0
    for paragraph in paragraphs:
        if len(paragraph) > config['min_p_length']:
            if ct_para % config['k'] == 0:
                # Wrap the last sentence of the paragraph with <keep> tags.
                # Assume new sentences start with a capital letter
                wrapped_paragraph = return_para_w_last_sentence_wrapped(paragraph, last_n_sentences=2)
                text_wrapped += wrapped_paragraph
            else:
                text_wrapped += paragraph + '\n\n'
            ct_para += 1
        else:
            if ct_sen % config['k'] == 0:
                wrapped_paragraph = return_para_w_last_sentence_wrapped(paragraph, last_n_sentences=1)
                text_wrapped += wrapped_paragraph
            else:
                text_wrapped += paragraph + '\n\n'
            ct_sen += 1

    return text_wrapped

def insert_keep_text(text, strategy: str ='eop', config: dict = None):
    if config is None:
        config = {}
    if strategy == 'eop':
        return insert_keep_text_at_eop(text, config)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

print(insert_keep_text(test_trace, 'eop', {}))
# verify that the wrapped text without keep tags is the same as the original text
def remove_keep_text(text):
    return text.replace('</keep>', '').replace('<keep>', '')

assert remove_keep_text(insert_keep_text(test_trace, 'eop', {})).strip() == test_trace.strip()
# Prompting
prompt = """
You are Deepseek, an expert reasoning Large Language Model. Your task is to assist in optimizing reasoning outputs by indicating which tokens are essential for future reasoning steps.

Currently, reasoning models often wrap their thought process within <think> and </think> tags. However, this approach can lead to unnecessarily long contexts, as many tokens included might not be crucial for subsequent reasoning.

To improve this, in addition to the existing <think> ... </think> tags (which denote the overall reasoning process), you will identify and mark specific tokens to preserve using <keep> and </keep> tags. These <keep> tags explicitly highlight tokens essential for the next steps of reasoning.

Your process:

1. Read the provided question and pre-existing reasoning trace.

2. Consider carefully: "If I continue this reasoning, which tokens must be retained?"

3. Clearly indicate the exact locations for inserting <keep> and </keep> tags around tokens necessary for further generation.

Important: Be thoughtful in choosing which tokens to wrap with <keep> tags. Strive for a balanced approach—avoid marking too few tokens, which may omit critical information, and avoid marking too many, which could reduce efficiency.

Output Format:

- Provide your selections clearly by indicating the paragraph number and the sentence number for each token or token group to retain. For example:

    - Paragraph 2, Sentence 3: <keep>essential tokens</keep>

- Ensure the text between <keep> and </keep> tags exactly matches the original text. Do not alter or paraphrase any tokens.

- Ensure the text between <keep> and </keep> tags is at least one complete sentence or phrase. You may include multiple sentences or phrases if they are closely related.

This method simulates efficient reasoning by focusing on essential tokens, optimizing context usage.
"""

def get_prompt_messsages(question, deepseek_thinking_trajectory):
    n_para = deepseek_thinking_trajectory.count('\n\n')
    user_prompt = f"""
    # Question:
    {question}

    # Deepseek Thinking Trajectory:
    {deepseek_thinking_trajectory}

    You must output at least {n_para // 5} blocks of kept text.
    """

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]
import os
from openai import OpenAI

client_nebius = OpenAI(
      base_url="https://api.studio.nebius.com/v1/",
      api_key=os.getenv('NEBIUS_API_KEY')
  )

def generate_response_nebius(client, messages, stream=False, verbose=False):
  response = client.chat.completions.create(
      model="deepseek-ai/DeepSeek-R1",
      max_tokens=16384,
      temperature=0.9,
      top_p=0.95,
      messages=messages,
      stream=stream
  )

  if stream:
      completion = ""
      for token in response:
        if hasattr(token, 'choices'):
            completion += token.choices[0].delta.content
            if verbose:
              print(token.choices[0].delta.content, end='', flush=True)
      return completion
  else:
    return response.choices[0].message.content
# Keep Text Parsing
import editdistance
from matplotlib import pyplot as plt
import numpy as np
with open('ex_keep_output.txt', 'r') as f:
    ex_keep_output = f.read()
with open('test_t.txt','r') as f:
    test_trace = f.read()

def extract_keep_text(ex_keep_output) -> list[str]:
    return re.findall(r': <keep>(.*?)</keep>', ex_keep_output)

keep_texts = extract_keep_text(ex_keep_output)

def check_keep_text(keep_text: list[str], test_trace: str) -> bool:
    # At most 10% of the texts can be missing
    missing_texts = 0
    for text in keep_text:
        if text not in test_trace:
            missing_texts += 1
    print(f'{missing_texts} / {len(keep_text)} = {missing_texts / len(keep_text)} missing texts')
    return missing_texts <= len(keep_text) * 0.1

check_keep_text(keep_texts, test_trace)

# Use word edit distance to check for close substrings
def word_edit_distance(text1: str, text2: str) -> int:
    return editdistance.eval(text1, text2)


def filter_keep_texts(keep_texts: list[str], test_trace: str) -> list[str]:
    # Filter out keep texts that aren't similar enough to the test trace
    valid_keep_texts = []
    for text in keep_texts:
        if text not in test_trace:
            # Find the closest match using word edit distance
            closest_match = min([test_trace[a:a+len(text)] for a in range(len(test_trace))], key=lambda x: word_edit_distance(text, x))

            # Only pick closest match if edit distance is more than 2 standard deviations from the mean
            std = np.std([word_edit_distance(text, test_trace[i:i+len(text)]) for i in range(len(test_trace))])
            mean = np.mean([word_edit_distance(text, test_trace[i:i+len(text)]) for i in range(len(test_trace))])
            deviation = abs(word_edit_distance(text, closest_match) - mean)
            if deviation > 9 * std:
                valid_keep_texts.append(closest_match)
        else:
            valid_keep_texts.append(text)
    proportion_kept = len(valid_keep_texts) / len(keep_texts)
    return valid_keep_texts, proportion_kept

valid_keep_texts, proportion_kept = filter_keep_texts(keep_texts, test_trace)
print(f'Proportion of keep texts kept: {proportion_kept}')
assert check_keep_text(valid_keep_texts, test_trace)
# Insert Keep tags

def insert_keep_tags(text: str, keep_texts: list[str]) -> str:
    for keep_text in keep_texts:
        # If the keep tag splits a word, we offset the keep tag to include the entire word
        # Find the first instance of the keep text in the text
        start_idx = text.find(keep_text)
        if start_idx != -1:
            # Find the last instance of the keep text in the text
            end_idx = start_idx + len(keep_text)
            # If the keep tag splits a word, we offset the keep tag to include the entire word
            for i in range(6):
                if text[start_idx-1].isalnum() and text[start_idx].isalnum():
                    start_idx -= 1
                else:
                    break
            for i in range(6):
                if text[end_idx].isalnum() and text[end_idx+1].isalnum():
                    end_idx += 1
                else:
                    break
        text = text[:start_idx] + f'<keep>{text[start_idx:end_idx]}</keep>' + text[end_idx:]
    return text

text_w_keep_tags = insert_keep_tags(test_trace, valid_keep_texts)
# ensure stripped text is the same as the original text
assert remove_keep_text(text_w_keep_tags).strip() == test_trace.strip()
# The whole output processing

def process_keep_additions(response: str, test_trace: str) -> str:
    keep_texts = extract_keep_text(response)
    valid_keep_texts, proportion_kept = filter_keep_texts(keep_texts, test_trace)
    print(f'Proportion of keep texts kept: {proportion_kept}')
    assert check_keep_text(valid_keep_texts, test_trace)
    text_w_keep_tags = insert_keep_tags(test_trace, valid_keep_texts)
    return text_w_keep_tags


from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as atqdm

### WORKING SYNCHRONOUS VERSION
# import os
# from tqdm import tqdm

# responses_dir = 'responses'
# questions = s1_df['question'].tolist()
# deepseek_thinking_trajectories = s1_df['deepseek_thinking_trajectory'].tolist()

# responses = []
# for i, (question, deepseek_thinking_trajectory) in tqdm(enumerate(zip(questions, deepseek_thinking_trajectories)), total=len(questions)):
#     if not os.path.exists(f'{responses_dir}/{i}/'):
#         os.makedirs(f'{responses_dir}/{i}/')
#     filename = f'{responses_dir}/{i}/first_run.txt'
#     if os.path.exists(filename):
#         with open(filename, 'r') as f:
#             responses.append(f.read())
#         continue
#     messages = get_prompt_messsages(question, deepseek_thinking_trajectory)
#     while True:
#         try:
#             response = generate_response_nebius(client_nebius, messages, stream=True)
#             break
#         except Exception as e:
#             print(f'Error generating response: {e}')
#             time.sleep(1)
#     responses.append(response)
#     with open(filename, 'w') as f:
#         f.write(response)
#     with open(f'{responses_dir}/{i}/original_ds_response.txt', 'w') as f:
#         f.write(deepseek_thinking_trajectory)

# tagged_responses = []
# for response in responses:
#     tagged_responses.append(process_keep_additions(response, deepseek_thinking_trajectory))

# s1_df['tagged_ds_response'] = tagged_responses
# sft_ds = s1_df[['question', 'deepseek_thinking_trajectory', 'solution', 'cot_type', 'source_type', 'tagged_ds_response']]
# sft_ds.to_parquet('sft_ds.parquet')



# Create async client
client_nebius_async = AsyncOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv('NEBIUS_API_KEY')
)

# Async version of generate_response_nebius
async def generate_response_nebius_async(client, messages):
    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=16384,
            temperature=0.9,
            top_p=0.95,
            messages=messages,
            stream=True
        )
        
        completion = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                completion += chunk.choices[0].delta.content
        
        return completion
    except Exception as e:
        print(f"Error in generate_response_nebius_async: {e}")
        raise e

# Update process_question to use the async version
async def process_question(id, question, deepseek_thinking_trajectory, semaphore, responses_dir):
    async with semaphore:
        uuid_index = id.split('--')[0]
        problem_source_id = id.split('--')[1]

        if not os.path.exists(f'{responses_dir}/{problem_source_id}/'):
            os.makedirs(f'{responses_dir}/{problem_source_id}/')
        
        filename = f'{responses_dir}/{problem_source_id}/{uuid_index}.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                response = f.read()
        else:
            messages = get_prompt_messsages(question, deepseek_thinking_trajectory)
            while True:
                try:
                    print(f'Generating response for completion {id}')
                    response = await generate_response_nebius_async(client_nebius_async, messages)
                    break
                except Exception as e:
                    print(f'Error generating response for completion {id}: {e} - retrying...')
                    await asyncio.sleep(1)
            print(f'Generated response for completion {id}')
            with open(filename, 'w') as f:
                f.write(response)
            # if not os.path.exists(f'{responses_dir}/{problem_source_id}/original_ds_response.txt'):
            #     with open(f'{responses_dir}/{problem_source_id}/original_ds_response.txt', 'w') as f:
            #         f.write(deepseek_thinking_trajectory)
        
        print(f'Returning response for completion {id}')
        return response, deepseek_thinking_trajectory

# Main async function
async def main():
    
    responses_dir = 'open_r1_traces'
    if not os.path.exists(responses_dir):
        os.makedirs(responses_dir)
    
    prob_df = pd.read_csv("open_r1_traces.csv")
    deepseek_thinking_trajectories = prob_df["reasoning"]
    questions = prob_df["problem"]
    ids = prob_df["id"]
    
    semaphore = asyncio.Semaphore(50)
    tasks = []
    
    for (question, deepseek_thinking_trajectory, id) in zip(questions, deepseek_thinking_trajectories, ids):
        task = asyncio.create_task(
            process_question(id, question, deepseek_thinking_trajectory, semaphore, responses_dir)
        )
        tasks.append(task)
    
    tagged_responses_and_trajectories = await atqdm.gather(*tasks)
    responses = [t[0] for t in tagged_responses_and_trajectories]
    trajectories = [t[1] for t in tagged_responses_and_trajectories]

    
    # tagged_responses = []
    # for response, trajectory in zip(responses, trajectories):
    #     tagged_responses.append(process_keep_additions(response, trajectory))
    
    df = pd.DataFrame({'question': questions, 'deepseek_thinking_trajectory': trajectories, 'response': responses})
    df.to_csv('open_r1_traces_with_tag_responses.csv', index=False)
# Run the async code
asyncio.run(main())