from pathlib import Path
from safe_subprocess import run
from generic_eval import main


LANG_NAME = "bash"
LANG_EXT = ".sh"

def eval_script(path: Path):
    # Capture output - will be generated regardless of success, fail, or syntax error
    p = run(["bash", path])
    if p.timeout:
        status = "Timeout"
    elif p.exit_code == 0:
        status = "OK"
    elif "syntax error" in p.stderr:
        status = "SyntaxError"
    else:
        status = "Exception"
    
    return {
        "status": status,
        "exit_code": p.exit_code,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }

if __name__ == "__main__":
    main(eval_script, LANG_NAME, LANG_EXT)
