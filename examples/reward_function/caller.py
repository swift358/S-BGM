# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
import math
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i):
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    datas = split_list(data,4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(4)]

        for future in as_completed(futures):
            print(future.result())

    for name in random_names:
        with open(name.replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))

    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


_HISTORY_QUESTIONS: List[str] = []


def _ngram_set(text: str, n: int = 3) -> set:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def _max_text_similarity(question: str, history: List[str]) -> float:
    current = _ngram_set(question)
    if not current or not history:
        return 0.0
    best = 0.0
    for old in history:
        old_set = _ngram_set(old)
        if not old_set:
            continue
        union = len(current | old_set)
        if union:
            best = max(best, len(current & old_set) / union)
    return best


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.0,
    file_path: str = "",
    uncertainty_sigma: float = 0.20,
    repetition_tau: float = 0.88,
    repetition_pool_size: int = 20000,
) -> List[Dict[str, float]]:
    """S-BGM Questioner reward.

    The external Solver service returns a consensus ratio p_hat in item["score"].
    We use r_Q(q)=max(0, exp(-((p_hat-0.5)^2)/(2*sigma^2)) - max(0, sim-tau)).
    """
    global _HISTORY_QUESTIONS
    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    scores = []
    for item in final_results:
        question = item.get("question", "")
        if not question:
            scores.append({"overall": -1.0, "format": 0.0, "accuracy": 0.0, "consensus_ratio": 0.0, "uncertainty": 0.0, "repetition": 0.0})
            continue
        p_hat = float(item.get("score", 0.0))
        p_hat = min(1.0, max(0.0, p_hat))
        r_unc = math.exp(-((p_hat - 0.5) ** 2) / (2.0 * uncertainty_sigma * uncertainty_sigma))
        similarity = _max_text_similarity(question, _HISTORY_QUESTIONS)
        r_rep = max(0.0, similarity - repetition_tau)
        role_reward = max(0.0, r_unc - r_rep)
        scores.append(
            {
                "overall": role_reward,
                "format": 1.0,
                "accuracy": 1.0 if item.get("answer") else 0.0,
                "consensus_ratio": p_hat,
                "uncertainty": r_unc,
                "repetition": r_rep,
            }
        )
    _HISTORY_QUESTIONS.extend(item.get("question", "") for item in final_results if item.get("question"))
    _HISTORY_QUESTIONS = _HISTORY_QUESTIONS[-repetition_pool_size:]
    return scores







