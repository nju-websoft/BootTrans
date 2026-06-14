#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-1.7B  # replace it with your local file path

python3 -m verl.trainer.main_trans \
    config=examples/qwen_lang.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
