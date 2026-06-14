      
import re
from typing import Any
import concurrent.futures
from containerized_eval import eval_string_script
from typing import Any, List, Dict
from pprint import pprint
import json
import random
from utils import extract_and_rename_java_functions
from tqdm import tqdm
import math
import multiprocessing as mp


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "using namespace std;"
    ],
    "java": ["import java.util.Arrays;"],
    "go"    : [
        # "math",
        # "strings",
        # "fmt",
        # "strconv",
        # "time",
        # "bytes",
        # "regexp",
        # "sort",
        # "math/rand",
        # "crypto/md5",
        "testing",
        "github.com/stretchr/testify/assert"
    ],
}

GO_INCLUDE = [
    "math",
    "strings",
    "fmt",
    "strconv",
    "time",
    "bytes",
    "regexp",
    "sort",
    "math/rand",
    "crypto/md5"
]

def markdown_codeblock_extract(new: str) -> str:
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


def post_process(new: str) -> str:
    try:
        extracted = markdown_codeblock_extract(new)
    except Exception as e:
        print(f"Failed to extract codeblock from {new}: {e}")
        extracted = new
    return extracted.strip()


def accuracy_reward(response: str, ground_truth: str, language: str, dataset: str):
    """计算单个响应的准确性奖励"""
    if dataset == 'humaneval':
        if language == 'java':
            response = response.split('public static void main')[0]
            prompt = response + '\n' + ground_truth.lstrip('    }')
        elif language == 'cpp':
            response = response.split('int main()')[0]
            prompt = response + '\n' + ground_truth.lstrip('}\n')
            if 'using namespace std;\n' not in prompt:
                prompt = '#include <iostream>\nusing namespace std;\n' + prompt
            if '#include <cassert>\n' not in  prompt:
                prompt = '#include <cassert>\n' + prompt
        else:
            prompt = response + '\n' + ground_truth
        ret = eval_string_script(language, prompt)
    
    elif dataset == 'humanevalx':
        if language == 'java':
            response = response.replace('public class Solution', 'class Solution')
            if 'public static void main' in response:
                response = response.split('public static void main')[0]
                prompt = response + '\n}\n' + ground_truth.replace('public class Main', 'public class Problem')
            else:
                prompt = response + '\n' + ground_truth.replace('public class Main', 'public class Problem')
        elif language == 'cpp':
            response = response.split('int main()')[0]
            prompt = response + '\n' + ground_truth.lstrip('}\n')
            if 'using namespace std;\n' not in prompt:
                prompt = '#include <iostream>\nusing namespace std;\n' + prompt
            if '#include <cassert>\n' not in  prompt:
                prompt = '#include <cassert>\n' + prompt
        
        elif language == 'go':
            test_script = response.split('func main')[0]
            if 'import' in test_script:
                split_result = test_script.split('func', 1)
                if len(split_result) >= 1:
                    test_script = 'func' + split_result[1]
            imp_lists = []
            for imp in IMPORT_HELPER.get(language, []):
                if imp not in test_script:
                    imp_lists.append(imp)
            for imp in GO_INCLUDE:
                if imp in test_script or imp in ground_truth:
                    imp_lists.append(imp)
            imp_statement = ""
            if len(imp_lists) != 0:
                tmp_str = '"\n    "'.join(imp_lists)
                imp_statement = f'import (\n    "{tmp_str}")\n'
            prompt = 'package main\n' + imp_statement + test_script.replace('package main\n', '') + '\n' + ground_truth

        
        else:
            prompt = response + '\n' + ground_truth
        print(prompt)
        
        ret = eval_string_script(language, prompt)
        print(ret['stdout'] + ' ---------------- ' + ret['status'] + '-------------' + ret['stderr'])
        print('*'*100)
    
    elif dataset == 'transcoder':
        if language == 'java':
            if 'class' in response:
                test_script = extract_and_rename_java_functions(response)
            else:
                method_pattern = re.compile(
                    r'(\s*(?:public|private|protected|static|final|abstract)\s+)*'  # 可选的修饰符 (组 1)
                    r'([a-zA-Z_$][a-zA-Z0-9_$.<>\[\]]*)\s+'                         # 返回类型 (组 2: 复杂的类型如 List<String>)
                    r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*'                                # 方法名 (组 3: 我们要提取和替换的)
                    r'(\([^)]*\))\s*\{',                                             # 参数列表 (组 4) 和起始 {
                    re.DOTALL
                )
                match = method_pattern.search(response)
                test_script = response
                if match:
                    original_name = match.group(3)
                    test_script = test_script.replace(original_name, 'f_filled')
                
            if test_script:
                test_script = ground_truth.replace('//TOFILL', test_script)
                test_script = re.sub(r"public class [^{]+\{", "public class Problem {", test_script)
            else:
                test_script = response
        
        elif language == 'cpp':
            pattern = r"int\s+main\s*\([^)]*\)\s*\{"
            match = re.search(pattern, response)
            if match:
                function_body = response[:match.start()]
            else:
                function_body = response
            pattern = r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{"
            method_names = re.findall(pattern, function_body)
            if method_names:
                function_body = function_body.replace(method_names[0], 'f_filled')
            test_script = ground_truth.replace('//TOFILL', function_body)

        elif language == 'python':
            pattern = r"\bdef\s+(\w+)\s*\([^)]*\)\s*:\s*"
            match = re.search(pattern, response)
            if match:
                function_body = response[match.start():]
                function_name = match.group(1)
                function_body = function_body.replace(function_name, 'f_filled')
                test_script = ground_truth.replace('#TOFILL', function_body)
            else:
                test_script = ground_truth  # 如果没有匹配到函数，保持原样
        
        for imp in IMPORT_HELPER.get(language, []):
            if imp not in test_script:
                test_script = imp + '\n' + test_script
            
        # print(test_script)
        ret = eval_string_script(language, test_script) 
        # print(ret['stdout'] + ' ---------------- ' + ret['status'] + '-------------' + ret['stderr'])
        # print('*'*100)
    
    return ret

def compute_single_score(reward_input: Dict[str, Any], dataset_name: str) -> Dict[str, float]:
    """为单个输入计算所有分数。"""
    response = reward_input["response"]
    language = reward_input.get("language", "java")

    eval_ret = accuracy_reward(response, reward_input[f"ground_truth"], language, dataset_name)
    
    if dataset_name == 'transcoder':
        return {
            'source_language': reward_input.get('source_language'),
            'target_language': language,
            'execution': eval_ret,
            'score': 1.0 if eval_ret.get('exit_code') == 0 and eval_ret.get("status") == 'OK' and ('#Results: 10, 10' in eval_ret.get("stdout") or '#Results:10, 10' in eval_ret.get("stdout")) else 0.0
        }
    else:
        return {
            'source_language': reward_input.get('source_language'),
            'target_language': language,
            'execution': eval_ret,
            'score': 1.0 if eval_ret.get('exit_code') == 0 and eval_ret.get("status") == 'OK' else 0.0
        }

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    dataset_name: str
) -> List[Dict[str, float]]:
    """使用 ProcessPoolExecutor 并行计算分数。"""
    # rets = []
    # for idx, item in enumerate(reward_inputs):
    #     ret = compute_single_score(item, dataset_name)
    #     rets.append(ret)
    
    # exit()
    max_workers = 10
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores_ordered = [None] * len(reward_inputs)   # 预分配，保证顺序
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(compute_single_score, item, dataset_name): idx
            for idx, item in enumerate(reward_inputs)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(reward_inputs), desc="Scoring"):
            idx = future_to_idx[future]            
            try:
                scores_ordered[idx] = future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                scores_ordered[idx] = {"status": "TimeoutError", "score": 0.0}
    return scores_ordered



class VLLMEngine:
    def __init__(self, name, num_gpus=1):
        import torch
        from vllm import LLM
        dtype = "auto"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"

        self.model = LLM(name, dtype=dtype, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
        self.tokenizer = self.model.get_tokenizer()

    def generate(self, convos: List[List[Dict[str, str]]], max_tokens: int, temperature: float, top_p: float, stop: Any = None):
        from vllm import SamplingParams
        formatted = []
        for convo in convos:
            contexts = [{"role": "user", "content": convo}]
            formatted.append(self.tokenizer.apply_chat_template(
                contexts, add_generation_prompt=True, tokenize=False))

        outputs = self.model.generate(
            formatted,
            SamplingParams(
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["```\n"]
                # stop=["```\n"] # R
            ),
        )

        return [post_process(o.outputs[0].text) for o in outputs]
        # return [
        #     (
        #         post_process(o.outputs[0].text),
        #         o.outputs[0].cumulative_logprob,
        #         len(o.outputs[0].token_ids),
        #     ) for o in outputs]

class Evaluator():
    def __init__(self, model_path, dataset_name, engine, num_gpus=1):
        self.model_path = model_path
        # if engine == "openai":
        #     self.engine = OpenAIEngine(name, endpoint)
        if engine == "vllm":
            self.engine = VLLMEngine(model_path, num_gpus)
        
        self.dataset_name = dataset_name
        if dataset_name == 'humanevalx':
            self.dataset_fname = 'data/validation_humanevalx.jsonl'
        else:
            self.dataset_fname = 'data/validation_transcoder.jsonl'
        
    
    def run_chunk(self, data_chunk, save_path, pbar):
        """处理一个切片，返回该切片指标"""
        prompts   = [l['source_prompt'] for l in data_chunk]
        gts       = [l['solution']       for l in data_chunk]
        src_langs = [l['source_language'] for l in data_chunk]
        tgt_langs = [l['target_language'] for l in data_chunk]

        # 1. 推理
        responses = self.translation(prompts, max_tokens=512, temperature=0,
                                    top_p=0.95, stop=None)
        # 2. 执行评测（带超时/异常保护）
        exe_rets = self.execution(responses, src_langs, tgt_langs, gts)

        # 3. 立即写盘 + 更新进度条
        with open(save_path, 'a', encoding='utf-8') as f:
            for ret in exe_rets:
                f.write(json.dumps(ret, ensure_ascii=False) + '\n')
        pbar.update(len(data_chunk))
        return exe_rets
    
    def translation(self, 
        prompts: List[str], 
        max_tokens: int, 
        temperature: float, 
        top_p, 
        stop
    ):
        
        outputs = self.engine.generate(
            prompts, max_tokens, temperature, top_p, stop)
        return outputs
    
    
    def execution(self, responses, source_languages, languages, test_cases):
        inputs = [{
            'response': r,
            'language': l,
            'ground_truth': gt,
            'source_language': sl
        } for r, l, gt, sl in zip(responses, languages, test_cases, source_languages)]
        # execution_ret = []
        # for i in inputs:
        #     execution_ret.append(compute_single_score(i))
        
        execution_ret = compute_score(inputs, self.dataset_name)
        return execution_ret
    
    def run(self, save_path, max_sample=None):
        data = []
        with open(self.dataset_fname, 'r') as f:
            for l in f:
                line = json.loads(l)
                data.append(line)
        
        if max_sample:
            sample_size = min(max_sample, len(data))
            data = random.sample(data, sample_size)
            # data = data[900:1500]
            
        prompts = [l['source_prompt'] for l in data]
        ground_truth = [l['solution'] for l in data]
        source_languages = [l['source_language'] for l in data]
        languages = [l['target_language'] for l in data]
        
        responses = self.translation(prompts, max_tokens=512, temperature=0, top_p=0.95, stop=None)
        
        execution_ret = self.execution(responses, source_languages, languages, ground_truth)
        
        scores = {}
        with open(save_path, 'w') as fw:
            for ret in execution_ret:
                key = f"{ret['source_language']}->{ret['target_language']}"
                scores[key] = scores.get(key, []) + [ret['score']]
                fw.write(json.dumps(ret) + '\n')
        
        for k, v in scores.items():
            print(k + ': ' + str(round(sum(v)/ len(v), 4) * 100) )

if __name__ == '__main__':
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'

    models = {
        'qwen_base': 'Qwen/Qwen3-1.7B',
    }
    
    def run_one(model_name, model_path, dataset_name):
        print('*' * 40 + model_name + '*' * 40)
        evaluator = Evaluator(model_path, engine='vllm', num_gpus=1, dataset_name=dataset_name)
        save_path = os.path.join('result', model_name + '_' + dataset_name + '.jsonl')
        evaluator.run(save_path)
    
    for model_name, model_path in models.items():
        run_one(model_name, model_path, 'transcoder')
        run_one(model_name, model_path, 'humanevalx')