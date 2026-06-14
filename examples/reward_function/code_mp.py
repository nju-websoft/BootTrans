
import re
import os
import concurrent.futures
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from containerized_eval import eval_string_script


# Metadata
REWARD_NAME = "code"
REWARD_TYPE = "lang"


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


def accuracy_reward(response: str, ground_truth: str, language: str, val_flag:bool=False) -> float:
    response = post_process(response)
    
    if not val_flag:
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
    
    elif val_flag:
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

        
        ret = eval_string_script(language, prompt)
  
    
    return 1.0 if ret['status'] == "OK" and ret.get('exit_code') == 0  else 0.0


def _compute_single_score(reward_input: dict[str, Any]) -> dict[str, float]:
    response = reward_input.get("response", "")
    target_language = reward_input.get("target_language", "")
    val_flag = reward_input.get("val", False)
    accuracy_score = accuracy_reward(
        response,
        reward_input["ground_truth"],
        language=target_language,
        val_flag=val_flag,
    )
    return {"overall": accuracy_score}


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    max_workers: int | None = None,
) -> list[dict[str, float]]:
    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)

    scores = [None] * len(reward_inputs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_single_score, ri): idx for idx, ri in enumerate(reward_inputs)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            scores[idx] = future.result()
    return scores
