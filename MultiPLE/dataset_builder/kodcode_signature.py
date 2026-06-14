
from transformers import AutoTokenizer
from pprint import pprint
from datasets import load_dataset
import json
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model_name = model_name.split('/')[-1]
dataset = load_dataset("KodCode/KodCode-Light-RL-10K")["train"]
output_fpath = 'datasets/kodcode_signature.jsonl'

prompt_template = '''
请你帮我生成这个函数的函数签名，我会给你这个函数的代码和对应的测试样例，需要你帮我重写一下函数签名，包含每个参数的数据类型以及函数的返回值。
### example
#### code
def count_distinct_elements(lst):\n    if not lst:\n        return 0\n    lst.sort()\n    count = 1\n    for i in range(1, len(lst)):\n        if lst[i] != lst[i-1]:\n            count += 1\n    return count\n
#### tests
from solution import count_distinct_elements\n\ndef test_no_elements():\n    assert count_distinct_elements([]) == 0\n\ndef test_single_element():\n    assert count_distinct_elements([1]) == 1\n\ndef test_all_distinct_elements():\n    assert count_distinct_elements([1, 2, 3, 4, 5]) == 5\n\ndef test_all_same_elements():\n    assert count_distinct_elements([2, 2, 2, 2, 2]) == 1\n\ndef test_mixed_elements():\n    assert count_distinct_elements([1, 2, 2, 3, 3, 4, 5, 6, 6]) == 6\n\ndef test_elements_with_negatives():\n    assert count_distinct_elements([-1, -1, 0, 1, 1, 2, 2]) == 4
#### signature
def count_distinct_elements(lst: List[int]) -> int:\n
### 请你帮我生成下面代码的函数签名，你的输出只需要有函数签名，不要有任何自然语言描述。
#### code
<source_code>
#### tests
<test_str>
#### signature
'''

from vllm import LLM, SamplingParams
llm = LLM(model=model_name, max_model_len=4096)

prompts = []
for line in dataset:
    pycode = line['solution']
    test_str = line['test']
    prompt = prompt_template.replace('<source_code>', pycode).replace('<test_str>', test_str)
    prompts.append(prompt)

sample_params = {
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.95,
    # "stop": ["\n\n"]
}
sample_params = SamplingParams(**sample_params)
outputs = llm.generate(prompts, sample_params)
assert len(dataset) == len(outputs)
with open(output_fpath, 'w') as fout:
    for i, (dataline, output) in enumerate(zip(dataset, outputs)):
        generation = output.outputs[0].text.strip()
        record = {
            'src_code': dataset[i]['solution'],
            'test': dataset[i]['test'],
            'tgt_signature': generation
        }
        fout.write(f"{json.dumps(record)}\n")

