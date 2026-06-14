import argparse
from sys import exit
import subprocess
from pathlib import Path
from generic_eval import main as gmain
import os
import shutil

def eval_script(path: Path):
    status = None
    stdout = None
    stderr = None
    exit_code = None
    try:
        go_file_path = path 
        
        module_dir = os.path.dirname(go_file_path)
        # 定义要复制的文件
        files_to_copy = ["go.mod", "go.sum"]

        # 遍历文件列表，将文件从 prev_temp 复制到 temp_dir
        for file_name in files_to_copy:
            source_file = os.path.join(module_dir, file_name)
            destination_file = os.path.join(module_dir, file_name)
            # print(destination_file)
            # 检查源文件是否存在
            if os.path.exists(source_file):
                shutil.copy2(source_file, destination_file)
                # print(f"Copied {file_name} to {temp_dir}")
            
        # exec_info = subprocess.run(["ls", "-al", module_dir], cwd=go_file_path)
        # exec_info = subprocess.run(["go",  "mod", "tidy"], cwd=module_dir)
        build = subprocess.run(
            ["go", "test", path],
            cwd=module_dir,
            timeout=30,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )

        stdout = build.stdout.decode("utf-8", errors="ignore")
        stderr = build.stderr.decode("utf-8", errors="ignore")
        exit_code = build.returncode
        # write to stderr just so that we can redirect stdout to a csv

        if "[setup failed]" in stdout or "[build failed]" in stdout:
            status = "SyntaxError"
        elif "FAIL" in stdout:
            status = "Exception"
        else:
            status = "OK"
    except subprocess.TimeoutExpired:
        status = "Timeout"

    return {
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }


if __name__ == "__main__":
    gmain(eval_script, 'Go', '.go')
