# BootTrans

Official reproduction guide for **BootTrans: Bootstrapping Code Translation with Weighted Multilanguage Exploration** (ACL 2026).

BootTrans studies reinforcement-learning-based code translation. This repository contains the training pipeline, code-execution reward, validation data, and evaluation script used to reproduce the main experimental workflow.

## 🧰 1. Environment Setup

### 1.1 Create Python Environment

We recommend using a clean Python environment. The project is tested with Python 3.9+.

```bash
conda create -n boottrans python=3.12 -y
conda activate boottrans

pip install -r requirements.txt
pip install -e .
```

The training code depends on `torch`, `transformers`, `vllm`, `ray`, and other packages listed in `requirements.txt`. Please install the CUDA-compatible versions of PyTorch and vLLM according to your local GPU driver and CUDA version if the default installation is not suitable.

### 1.2 Install MultiPL-E Evaluation Package

BootTrans uses the MultiPL-E execution framework for code evaluation.

```bash
cd MultiPLE/evaluation
pip install -e .
cd ../..
```

### 1.3 Prepare Java Dependencies

Some Java evaluation cases require the bundled JavaFX environment.

```bash
unzip ./envs/javafx.zip -d ./envs/
```

The repository also provides `envs/javatuples-1.2.jar` for Java-related execution.

## 📦 2. Data

The released data files are placed under `data/`.

| Split | File | Description |
| --- | --- | --- |
| Training | `data/kodcode.jsonl` | Training data for BootTrans. |
| Validation | `data/validation_humanevalx.jsonl` | HumanEval-X validation set. |
| Validation | `data/validation_transcoder.jsonl` | TransCoder validation set. |

Each JSONL sample follows the fields consumed by the training and evaluation scripts, including source prompt, solution, source language, and target language.


## 🏋️ 3. Training

The default training entry point is:

```bash
bash ./examples/boottrans.sh
```

By default, the script uses `Qwen/Qwen3-1.7B` as the base model:

```bash
MODEL_PATH=Qwen/Qwen3-1.7B
```

To reproduce experiments with another base model, modify both:

1. `MODEL_PATH` in `examples/boottrans.sh`
2. `worker.actor.model.model_path` in `examples/qwen_lang.yaml`

The main configuration file is `examples/qwen_lang.yaml`. Important fields include:

| Field | Meaning |
| --- | --- |
| `data.train_files` | Training JSONL file. |
| `data.val_files` | Validation JSONL file used during training. |
| `algorithm.adv_estimator` | Advantage estimator; BootTrans uses `grpo_lang`. |
| `worker.rollout.n` | Number of sampled responses per prompt. |
| `worker.rollout.tensor_parallel_size` | Tensor parallel size for rollout. |
| `trainer.n_gpus_per_node` | Number of GPUs used on each node. |
| `trainer.total_epochs` | Number of training epochs. |
| `trainer.save_freq` | Checkpoint saving frequency. |

For a single-node run, set `trainer.n_gpus_per_node` and `worker.rollout.tensor_parallel_size` according to the available GPUs. The default configuration uses 2 GPUs.

## 📊 4. Evaluation

Evaluation is implemented in `evaluation/autochatmodel.py`. The script loads a model with vLLM, generates translations for both validation sets, executes the generated programs, and reports execution accuracy by language pair.

### ⚙️ 4.1 Configure the Model

Edit the `models` dictionary in `evaluation/autochatmodel.py`:

```python
models = {
    "qwen_base": "Qwen/Qwen3-1.7B",
}
```

The key is the output name, and the value is the Hugging Face model ID or a local checkpoint path.

### ▶️ 4.2 Run Evaluation

Run the evaluation script from the repository root:

```bash
python ./evaluation/autochatmodel.py
```

The script evaluates:

```text
data/validation_transcoder.jsonl
data/validation_humanevalx.jsonl
```

and writes JSONL results to:

```text
result/<model_name>_transcoder.jsonl
result/<model_name>_humanevalx.jsonl
```


## 📝 5. Notes

- Code execution evaluation requires the corresponding language runtimes to be available in the environment.
- If using a local checkpoint saved by training, set the checkpoint path in `evaluation/autochatmodel.py`.

## 📚 Citation

If you use this repository, please cite:

```bibtex
@inproceedings{boottrans2026,
  title = {Bootstrapping Code Translation with Weighted Multilanguage Exploration},
  booktitle = {ACL},
  year = {2026}
}
```
