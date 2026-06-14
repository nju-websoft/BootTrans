from pathlib import Path
import tempfile
from safe_subprocess import run
import os

LANG_NAME = "Scala"
LANG_EXT = ".scala"
SCALA_BIN = "<SCALA_BIN_HERE>"
def eval_script(path: Path):
    content = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    with tempfile.TemporaryDirectory() as outdir:
        # Each Scala file contains the class with same name `JAVA_CLASS_NAME`
        # Hence, scalac will same JAVA_CLASS_NAME.class file for each problem
        # Write class for each problem to a different temp dir
        
        # content = open(path).read()
        new_path = os.path.join(outdir, "Problem.scala")
        with open(new_path, 'w', encoding='utf-8') as new_file:
            new_file.write(content)
        
        # print([SCALA_BIN + "scalac", "-d", outdir, new_path])
        build = run(["scalac", "-d", outdir, new_path], timeout_seconds=45)
        # exit()
        if build.exit_code != 0:
            # Well, it's a compile error. May be a type error or
            # something. But, why break the set convention
            return {
                "status": "SyntaxError",
                "exit_code": build.exit_code,
                "stdout": build.stdout,
                "stderr": build.stderr,
            }
        
        # print(["scala", "-cp", f"{outdir}", "Problem"])
        # "Problem" is the name of the class we emit.
        r = run([SCALA_BIN + "scala", "-cp", f"{outdir}", "Problem"])
        if r.timeout:
            status = "Timeout"
        elif r.exit_code == 0 and r.stderr == "":
            status = "OK"
        else:
            # Well, it's a panic
            status = "Exception"
    return {
        "status": status,
        "exit_code": r.exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
