from datasets import load_dataset
from generic_translator import translate_tests, translate_prompt, PromptVisitor
import json
import ast
# from generic_translator import PromptVisitor
import re

def main():
    kod_dataset = load_dataset("KodCode/KodCode-Light-RL-10K")["train"]
    tgt_langs = ['py', 'java', 'cpp', 'js', 'go']
    # tgt_langs = ['py', ]
    full_mapping = {
        'java': 'java', 
        'cpp': 'cpp',
        'js': 'javascript',
        'go': 'go',
        'py': 'python',
    }
    signatures = []
    base_dir = './datasets/kodcode/'
    output_file = f'{base_dir}kodcode_mix_train_all.jsonl'
    signature_file = f'{base_dir}kodcode_signature_py.jsonl'
    
    with open(signature_file, 'r') as f:
        for l in f:
            line = json.loads(l)
            signatures.append(line)


    with open(output_file, 'w') as f:
        for i, (line, signature) in enumerate(zip(kod_dataset, signatures)):
            
            entry_point = line['test'].split('\n')[0].replace('from solution import ', '')
            tests = line['test'].replace(f'from solution import {entry_point}', '')
            # prompt = before_2nd_triple_quote(line['solution']) + '\n'
            # prompt = 'def findThreeSum(nums: List[int], k: int) -> bool:\n    """pass"""\n'
            prompt = signature['tgt_signature'] if '"""' in signature['tgt_signature'] else signature['tgt_signature']  +'\n    """None"""\n'
            # prompt = line['solution'] + '\n'
            
            translations_prompts = {}
            for lang in tgt_langs:
                translator = __import__(f'humaneval_to_{lang}').Translator()
                
                translated_prompt = translate_prompt(
                    translator, 'keep', prompt,  "", added_canonical="")

                translated_tests = translate_tests(
                    translator,
                    tests,
                    entry_point,
                    'test.py',
                    False
                )
                
                if translated_prompt is not None and translated_tests is not None:
                    translations_prompts[lang] = (translated_prompt, translated_tests)
                

            # 如果翻译成功
            if len(translations_prompts) == len(tgt_langs):
                solution = line['solution']
                # 删除doc string
                try:
                    prompt_ast = ast.parse(solution)
                    prompt_visitor = PromptVisitor(translator)
                    prompt_visitor.visit(prompt_ast)
                    cleaned_py = solution.replace('"""' + prompt_visitor.description + '"""', '')
                except:
                    cleaned_py = solution
                
                # 删除注释
                cleaned_py = re.sub(r'(?m)^\s*#.*\n?', '', cleaned_py)
                cleaned_py = re.sub(r'\n{3,}', '\n\n', cleaned_py).strip()

                # line['python_prompt'] = cleaned_py
                # line['python_test'] = line['test']
                line['source_prompt'] = cleaned_py
                for lang, value in translations_prompts.items():
                    tgt_lang = full_mapping[lang]
                    translated_prompt, translated_tests = value
                    
                    line[f'{tgt_lang}_prompt'] = translated_prompt
                    line[f'{tgt_lang}_test'] = translated_tests
    #             line[f'tranlation_prompt'] = f'''```python
    # {cleaned_py}
    # ```
    # The translated {tgt_lang} code should be:
    # ```{tgt_lang}
    # {translated_prompt}
    # ```'''
                
                # 写入文件
                f.write(json.dumps(line) + '\n')

main()