# Low Resource Program Language Translation

## 实验步骤
### 准备目标语言的函数签名
首先要准备目标语言的函数签名，用于测试。
需要做的事情是先改dataset_builder下面的humaneval_to_X.py文件，在Translator类下面新增一个translate_function_signature方法，在translate_prompt的基础上删除问题描述，只保留函数签名。
然后更改all_prepare_prompts.py文件中的LANGS变量，将目标语言添加进去。

然后就可以开始构造数据集了。
命令如下：
```bash
python all_prepare_prompts.py
```


### 准备源语言代码
这里我直接写到了autochatmodel.py的load_{humaneval/mbpp}_python_dataset函数，此后在ChatModel的translations方法中调用，加载数据集，并拼接。

此后，记得改一下/data1/yhwu/github_project/MultiPL-E/chat-templates/translation.yaml文件中的目标语言。

预测的代码如下：
```bash
python autochatmodel.py \
    --temperature 0 \
    --completion-limit 1 \
    --name /data1/shares/Qwen2.5-7B-Instruct \
    --lang java \
    --root-dataset mbpp \
    --engine vllm \
    --chat-template /data1/yhwu/github_project/MultiPL-E/chat-templates/translation.yaml \
    --use-local \
    --dataset /data1/yhwu/github_project/MultiPL-E/translation_prompts/mbpp-java-keep.jsonl
```

这里有几个参数要注意：
--completion-limit 1表示只生成一个补全；
--name表示使用的模型，这里使用的是Qwen2.5-7B-Instruct；
--lang表示目标语言，这里使用的是d；
--root-dataset表示数据集是mbpp/humaneval；
--chat-template表示使用哪个chat模板，这里使用的是translation.yaml；
--use-local表示使用本地构建的函数签名；
--dataset表示本地构造数据集的路径。

### 评估
首先需要安装对应语言的工具，对于Dlang，每次需要source一下。

评估比较简单运行
```bash
python ./evaluation/src/command_line.py \
    --output_dir /data1/yhwu/github_project/MultiPL-E/results/temp \
    --dir /data1/yhwu/github_project/MultiPL-E/humaneval-py-keep-_data1_shares_Qwen2.5_7B_Instruct-0.0-reworded \
    --test_results_path /data1/yhwu/github_project/MultiPL-E/results/generation_qwen7b_instruct/run-humaneval-py.json
```

--output_dir 暂时写入的文件，默认写这个就行
--dir 上一步预测的结果的文件夹路径
--test_results_path 最终测试结果的输出路径

### 还有几个实用的终端命令
#### 关于json.gz终端显示
zcat filename.json.gz
#### 关于jq的使用
对于json文件，jq -r '.[0].program' 表示查看第0个元素的program字段
#### 关于json文件中换行的可视化
使用管道
awk '{ gsub(/\\n/,"\n"); print }'




# 采样
```
python autochatmodel.py \
    --temperature 0 \
    --max-tokens 2048 \
    --completion-limit 1 \
    --name /data1/yhwu/github_project/EasyR1/checkpoints/easy_r1/qwen2_5_7b_java_grpo/global_step_11/actor/huggingface \
    --src-lang python \
    --lang cpp \
    --root-dataset humanevalx \
    --engine vllm \
    --chat-template /data1/yhwu/github_project/MultiPL-E/chat-templates/translation_grpo.yaml \
    --use-local \
    --dataset /data1/yhwu/github_project/MultiPL-E/translation_prompts/mbpp-cpp-keep.jsonl \
    --output-dir /data1/yhwu/github_project/MultiPL-E/prilimilary_results/grpo/

- humanevalx用下面的脚本，可以支持多语言版本，保证dataset中的lang等于lang(目标语言)
python autochatmodel.py \
    --temperature 0 \
    --max-tokens 2048 \
    --completion-limit 1 \
    --name /data1/shares/Qwen2.5-7B-Instruct \
    --src-lang python \
    --lang cpp \
    --root-dataset humanevalx \
    --engine vllm \
    --chat-template /data1/yhwu/github_project/MultiPL-E/chat-templates/translation_grpo.yaml \
    --use-local \
    --dataset /data1/yhwu/github_project/research/transop/data/humanevalx/processed/cpp.jsonl \
    --output-dir /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen25_instruct_humanevalx/

```

# 评估
python ./evaluation/src/main.py \
    --dir /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen25_instruct_humanevalx/humanevalx-python-cpp.jsonl \
    --output-dir /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen25_instruct_humanevalx/execution/humanevalx-python-cpp/ \
    --lang cpp


# 计算pass
python pass_k.py /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen2_5_7b_mix_grpo_lang/execution/humanevalx-cpp-python /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen2_5_7b_mix_grpo_lang/execution/humanevalx-java-cpp /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen2_5_7b_mix_grpo_lang/execution/humanevalx-java-python /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen2_5_7b_mix_grpo_lang/execution/humanevalx-python-cpp

python pass_k.py /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen2_5_7b_mix_grpo/execution/humanevalx-python-cpp /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen25_instruct_humanevalx/execution/humanevalx-python-cpp/

python pass_k.py /data1/yhwu/github_project/MultiPL-E/prilimilary_results/qwen25_instruct/execution/mbpp-julia/