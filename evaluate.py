import argparse
import ast
import gzip
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import requests
import tqdm
from tqdm import tqdm
from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    'build/tree_sitter.so',

    # Include one or more languages
    ['tree-sitter-python', 'tree-sitter-java', 'tree-sitter-cpp', 'tree-sitter-c-sharp']
)

TS_LANG = {
    "python": Language('build/tree_sitter.so', 'python'),
    "java": Language('build/tree_sitter.so', 'java'),
    "cpp": Language('build/tree_sitter.so', 'cpp'),
    "csharp": Language('build/tree_sitter.so', 'c_sharp')
}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


class ASTVisitor:

    def __init__(self, with_ndtypes=False):
        self.with_ndtypes = with_ndtypes
        self.stack = []
        self.ndtypes = []

    def enter(self, node) -> bool:
        return True

    def leave(self, node):
        pass

    def enter_leaf(self, node):
        pass

    def on_enter(self, node) -> bool:
        # print("on enter ", node.type)
        if self.with_ndtypes:
            self.ndtypes.append((node.start_byte, True, node.type))
        enter_fn = getattr(self, "enter_%s" % node.type, self.enter)
        r = enter_fn(node)
        if node.child_count == 0:
            self.enter_leaf(node)
        self.stack.append(node.type)
        return r

    def on_leave(self, node):
        assert self.stack.pop() == node.type
        leave_fn = getattr(self, "leave_%s" % node.type, self.leave)
        r = leave_fn(node)
        # print("on leave ", node.type)
        if self.with_ndtypes:
            self.ndtypes.append((node.end_byte, False, node.type))
        return r

    def walk(self, root_node):
        if root_node is None:
            return

        cursor = root_node.walk()
        has_next = True

        while has_next:
            current_node = cursor.node

            # Step 1: Try to go to next child if we continue the subtree
            if self.on_enter(current_node):
                has_next = cursor.goto_first_child()
            else:
                has_next = False

            # Step 2: Try to go to next sibling
            if not has_next:
                self.on_leave(current_node)
                has_next = cursor.goto_next_sibling()

            # Step 3: Go up until sibling exists
            while not has_next and cursor.goto_parent():
                self.on_leave(cursor.node)  # We will never return to this specific parent
                has_next = cursor.goto_next_sibling()

    def __call__(self, root_node):
        return self.walk(root_node)


class ErrorCheckVisitor(ASTVisitor):
    def __init__(self):
        super().__init__()
        self.error_cnt = 0

    def enter_ERROR(self, node):
        if node.text.decode("utf-8") != ";":
            self.error_cnt += 1


class ExecOutcome(Enum):
    PASSED = "PASSED"  # code executes and output matches expected output
    WRONG_ANSWER = "WRONG_ANSWER"  # code executes and output does NOT matches expected output
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"  # code executes and didn't exit in time, output is ignored in this case
    RUNTIME_ERROR = "RUNTIME_ERROR"  # code failed to execute (crashed)
    COMPILATION_ERROR = "COMPILATION_ERROR"  # code failed to compile
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"  # code exceeded memory limit during execution


@dataclass
class ExtendedUnittest:
    input: str
    output: List[str] = field(default_factory=list)
    result: Optional[str] = None
    exec_outcome: Optional[ExecOutcome] = None

    def json(self):
        _json = self.__dict__
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.name

        return _json

    @classmethod
    def from_json(cls, _json):
        return cls(
            input=_json.get("input", ""),
            output=_json.get("output", list()),
            result=_json.get("result", None),
            exec_outcome=_json.get("exec_outcome", None), )


class APICommunication:
    _session: requests.Session

    def __init__(self, server_url: str = "http://localhost:5000"):
        self._session = requests.Session()
        self.execute_code_url = f"{server_url}/api/execute_code"
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def get_runtimes(self):
        return self._session.get(self.get_runtimes_url).json()

    def execute_code(
        self,
        language: str,
        source_code: str,
        unittests: List[dict],
        limits: Optional[dict] = None,
        block_network: bool = True,
        stop_on_first_fail: bool = True,
        use_sanitizer: bool = False,
        compiler_program_name: Optional[str] = None,
        compiler_flags: Optional[str] = None,
        interpreter_cmd: Optional[str] = None,
        interpreter_flags: Optional[str] = None,
        sample_id: Optional[int] = None,
        task_id: Union[str, int, None] = None, ) -> Tuple[List[ExtendedUnittest], Optional[int], Union[str, int, None]]:
        if language is None:
            raise ValueError("EmptyLanguage")

        if source_code is None:
            raise ValueError("EmptySourceCode")

        if unittests is None or len(unittests) == 0:
            raise ValueError("EmptyUnittest")

        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer, )
        try:
            json_response = self._session.post(
                self.execute_code_url, json=request_body, headers={"Content-Type": "application/json"}, ).json()
        except requests.exceptions.JSONDecodeError:
            json_response = {
                "task_id": task_id,
                "data": [{"exec_outcome": "COMPILATION_ERROR", "result": "", "passed": False}]
            }

        if "data" not in json_response:
            return json_response, sample_id, task_id

        return (json_response["data"], sample_id, task_id,)


LANG_TO_COMPILER = {
    "cpp": "GNU C++17", "csharp": "Mono C#", "java": "Java 17", "python": "PyPy 3"
}

execeval: APICommunication = None


def run_test(problem, completion):
    global execeval
    assert problem['task_id'] == completion['task_id']
    code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
    result = execeval.execute_code(
        LANG_TO_COMPILER[problem['lang']], code, problem['unit_tests'], task_id=problem['task_id']
    )[0]
    for o in result:
        if o['result'] is not None and len(o['result']) > 1000:
            o['result'] = o['result'][:1000]
    return result, all(o['exec_outcome'] == 'PASSED' for o in result)


def check_syntax(code):
    parser = Parser()
    parser.set_language(TS_LANG["python"])
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0


def get_function_call_params(node):
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
    return (positional_args, keyword_args)


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
    parser.add_argument("data_path", type=str)
    parser.add_argument("completion_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    global execeval
    execeval = APICommunication(server_url=f"http://localhost:{args.port}")

    completions = {completion["task_id"]: completion for completion in stream_jsonl(args.completion_path)}
    pass_cnt, total = 0, 0
    results = []
    for problem in tqdm(stream_jsonl(args.data_path)):
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
                    result, passed = run_test(problem, completion)
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
