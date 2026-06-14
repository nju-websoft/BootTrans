import os
import tempfile
from safe_subprocess import run
from pathlib import Path
from generic_eval import main

LANG_NAME = "Java"
LANG_EXT = ".java"

#Following files have problems:
#137, 
#22: Any
#148: Elipsis

def eval_script(path: Path):

    sys_env = os.environ.copy()
    javatuples_path = Path("./envs/javatuples-1.2.jar")
    module_path = "./envs/javafx-sdk-21.0.5/lib"
    sys_env["CLASSPATH"] =  f"{javatuples_path}"

    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    with tempfile.TemporaryDirectory() as outdir:
        #Each Java file contains the class with same name `JAVA_CLASS_NAME`
        #Hence, javac will same JAVA_CLASS_NAME.class file for each problem
        #Write class for each problem to a different temp dir
        #Use UTF8 encoding with javac
        
        # result = run(["javac", "-encoding", "UTF8", "-d", outdir, path], env=sys_env)
        new_path = os.path.join(outdir, "Problem.java")
        with open(new_path, 'w', encoding='utf-8') as new_file:
            new_file.write(content)
        
        # result = run(["javac", "-encoding", "UTF8", "-d", outdir, new_path], env=sys_env)
        result = run(["javac", "--module-path", module_path, "--add-modules", "javafx.controls", "-cp", f"{javatuples_path}", "-encoding", "UTF8", "-d", outdir, new_path], env=sys_env, timeout_seconds=5)
         
        if result.exit_code != 0:
            # Well, it's a compile error. May be a type error or
            # something. But, why break the set convention
            status = "SyntaxError"
        else:
            
            # result = run(["java", "-ea", "-cp", f"{outdir}:{javatuples_path}", "Problem"], env = sys_env)
            result = run(["java", "--module-path", module_path, "--add-modules", "javafx.controls", "-ea", "-cp", f"{outdir}:{javatuples_path}", "Problem"], env = sys_env, timeout_seconds=5)
            
            if result.timeout:
                status = "Timeout"
            elif result.exit_code == 0:
                status = "OK"
            else:
                status = "Exception"

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

if __name__ == "__main__":
    main(eval_script, LANG_NAME, LANG_EXT)
