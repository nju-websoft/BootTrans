from pathlib import Path
from containerized_eval import eval_string_script
import itertools
import argparse
from tqdm import tqdm
import gzip
import json

def get_test_results_json_path(output_dir: Path, problem_json_path: Path, input_dir: Path) -> Path:
    suffixes = ".results.json.gz" if problem_json_path.suffix == ".gz" else ".results.json"
    problem_name = problem_json_path.name[:-(len(".json.gz") if problem_json_path.suffix == ".gz" else len(".json"))]
    if input_dir:
        return output_dir / (problem_json_path.relative_to(input_dir).parent / (problem_name + suffixes))
    return output_dir / (problem_name + suffixes)


def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode) 


def evaluate_problem(output_dir: Path, problem_json_path: Path, input_dir: Path = None):
    with open_json(problem_json_path, "r") as f:
        problem = json.load(f)

    # Do not create a blank .results.json file if there are no completions ready.
    if len(problem["completions"]) == 0:
        return

    index = 0
    
    completion = problem["completions"][index]
    # assert "rm -" not in program
    
    if problem["language"] == "sh":
        program = problem["prompt"] + completion + '\n' + problem["tests"][1:]
    elif problem["language"] == "d":
        program = problem["prompt"] + completion.split("void main()")[0] + '\n' + problem["tests"]
    elif problem["language"] == "java":
        if "public static void main" in completion:
            prefix = "public static void main"
            program = problem["prompt"] + completion.split(prefix)[0].rstrip()[:-1] + '\n' + problem["tests"]
        elif "void main" not in completion:
            prefix = "}"

            # assert completion.endswith("}\n}")
            program = problem["prompt"] + completion.rstrip("}\n}") + '\n' + problem["tests"]
        else:
            print(completion)
            assert 0, "here"
    else:
        program = problem["prompt"] + completion + '\n' + problem["tests"]
    
    result_dict = eval_string_script(problem["language"], program)
    result_dict["name"] = problem["name"]
    return result_dict
    # return "None"

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--output_dir", type=Path,
        help="Directory to store results in. Ignored when using a --job-file")
    args.add_argument("--dir", type=str, help="The directory to evaluate")
    args.add_argument("--recursive", action="store_true", help="Read all files under each directory, recursively. Only works with --dir.")
    args.add_argument("--test_results_path", help="Run save dir")
    
    args = args.parse_args()

    files = [ p for p in itertools.chain(Path(args.dir).glob("**/*.json" if args.recursive else "*.json"), \
                                             Path(args.dir).glob("**/*.json.gz" if args.recursive else "*.json.gz")) \
                    if not p.name.endswith(".results.json") and not p.name.endswith(".results.json.gz")  ] 

    test_results = []
    for file in tqdm(files):
        test_results.append(evaluate_problem(args.output_dir, file, args.dir))

    with open(args.test_results_path, "w") as f:
        f.write(json.dumps(test_results, indent=2))


main()