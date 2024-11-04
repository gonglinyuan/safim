import argparse
import ast
import json
import re

import tqdm
from tqdm import tqdm

from ast_utils import ErrorCheckVisitor, get_parser
from data_utils import load_dataset, stream_jsonl
from exec_utils import build_execeval, run_test


def check_syntax(code):
    parser = get_parser("python")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0


def get_function_call_params(node):
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
    return positional_args, keyword_args


def function_calls_match(call1, call2):
    params1 = get_function_call_params(call1)
    params2 = get_function_call_params(call2)
    return params1 == params2


def syntax_match(code1, code2, lang):
    code1 = re.sub(r'\s+', '', code1).strip()
    code2 = re.sub(r'\s+', '', code2).strip()
    if lang == "python":
        try:
            tree1 = ast.parse(code1, mode='eval')
            tree2 = ast.parse(code2, mode='eval')

            if isinstance(tree1.body, ast.Call) and isinstance(tree2.body, ast.Call):
                return function_calls_match(tree1.body, tree2.body)
        except:
            pass  # If parsing fails, fall back to simple string comparison

    return code1 == code2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("completion_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--compiler", type=str)
    args = parser.parse_args()

    build_execeval(args)

    completions = {completion["task_id"]: completion for completion in stream_jsonl(args.completion_path)}
    pass_cnt, total = 0, 0
    results = []
    for problem in tqdm(load_dataset(args.completion_type, args.lang)):
        if problem["task_id"] not in completions:
            result = "EMPTY"
            passed = False
        else:
            completion = completions[problem["task_id"]]
            if "unit_tests" in problem and problem["unit_tests"]:
                if completion['completion'] == problem["ground_truth"]:
                    result = "PASSED"
                    passed = True
                else:
                    result, passed = run_test(problem, completion, args.compiler)
            else:
                if syntax_match(completion['completion'], problem["ground_truth"], problem["lang"]):
                    result = "EXACT_MATCH"
                    passed = True
                else:
                    result = "WRONG_ANSWER"
                    passed = False
        if not completion['completion'].strip() and not passed:
            result = "EMPTY"
        if problem["lang"] == "python" and not passed:
            full_code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
            if "unit_tests" in problem and not check_syntax(full_code):
                result = "COMPILATION_ERROR"
        pass_cnt += int(passed)
        total += 1
        results.append(
            {
                "task_id": problem["task_id"], "result": result, "passed": passed, "check_result": 0
            }
        )

    print(f"Pass {pass_cnt} / Total {total}")
    print(f"Pass@1: {pass_cnt / total * 100 :.04f}%")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == '__main__':
    main()
