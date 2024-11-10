from typing import Iterable

import jinja2

from ast_utils import ErrorCheckVisitor, get_parser
from model_utils import ModelWrapper


def truncate_to_first_line(code, add_line_break=False):
    lines = code.splitlines()
    for line in lines:
        if line.strip():
            if not add_line_break:
                return line
            else:
                return line + "\n"
    return ""


COMPLETION_PLACEHOLDER = {
    "python": "# TODO: Your code here",
    "java": "/* TODO: Your code here */",
    "cpp": "/* TODO: Your code here */",
    "csharp": "/* TODO: Your code here */",
}

FEW_SHOT_PROMPTS, FEW_SHOT_ANSWERS = {}, {}

FEW_SHOT_PROMPTS[("statement", "python")] = """\
Complete the code in python to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

t = int(input())
for _ in range(t):
    a, b = map(int, input().split())
    # TODO: Your code here
    print(c)
"""

FEW_SHOT_ANSWERS[("statement", "python")] = "c = a + b"

FEW_SHOT_PROMPTS[("statement", "java")] = """\
Complete the code in java to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            int a = sc.nextInt();
            int b = sc.nextInt();
            /* TODO: Your code here */
            System.out.println(c);
        }
        sc.close();
    }
}
"""

FEW_SHOT_ANSWERS[("statement", "java")] = "int c = a + b;"

FEW_SHOT_PROMPTS[("statement", "cpp")] = """\
Complete the code in cpp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

#include <iostream>
using namespace std;

int main() {
    int t, a, b, c;
    cin >> t;
    for (int i = 0; i < t; i++) {
        cin >> a >> b;
        /* TODO: Your code here */
        cout << c << endl;
    }
    return 0;
}
"""

FEW_SHOT_ANSWERS[("statement", "cpp")] = "c = a + b;"

FEW_SHOT_PROMPTS[("statement", "csharp")] = """\
Complete the code in csharp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

using System;

class Program {
    static void Main() {
        int t = int.Parse(Console.ReadLine());
        for (int i = 0; i < t; i++) {
            string[] inputs = Console.ReadLine().Split(' ');
            int a = int.Parse(inputs[0]);
            int b = int.Parse(inputs[1]);
            /* TODO: Your code here */
            Console.WriteLine(c);
        }
    }
}
"""

FEW_SHOT_ANSWERS[("statement", "csharp")] = "int c = a + b;"

FEW_SHOT_PROMPTS[("block", "python")] = """\
Complete the code in python to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

t = int(input())
for _ in range(t):
    # TODO: Your code here
"""

FEW_SHOT_ANSWERS[("block", "python")] = """\
a, b = map(int, input().split())
    c = a + b
    print(c)
"""

FEW_SHOT_PROMPTS[("block", "java")] = """\
Complete the code in java to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            /* TODO: Your code here */
        }
        sc.close();
    }
}
"""

FEW_SHOT_ANSWERS[("block", "java")] = """\
int a = sc.nextInt();
            int b = sc.nextInt();
            int c = a + b;
            System.out.println(c);
"""

FEW_SHOT_PROMPTS[("block", "cpp")] = """\
Complete the code in cpp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

#include <iostream>
using namespace std;

int main() {
    int t, a, b, c;
    cin >> t;
    for (int i = 0; i < t; i++) {
        /* TODO: Your code here */
    }
    return 0;
}
"""

FEW_SHOT_ANSWERS[("block", "cpp")] = """\
cin >> a >> b;
        c = a + b;
        cout << c << endl;
"""

FEW_SHOT_PROMPTS[("block", "csharp")] = """\
Complete the code in csharp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

using System;

class Program {
    static void Main() {
        int t = int.Parse(Console.ReadLine());
        for (int i = 0; i < t; i++) {
            /* TODO: Your code here */
        }
    }
}
"""

FEW_SHOT_ANSWERS[("block", "csharp")] = """\
string[] inputs = Console.ReadLine().Split(' ');
            int a = int.Parse(inputs[0]);
            int b = int.Parse(inputs[1]);
            Console.WriteLine(c);
"""

FEW_SHOT_PROMPTS[("control", "python")] = """\
Complete the code in python to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

t = int(input())
for # TODO: Your code here
    a, b = map(int, input().split())
    c = a + b
    print(c)
"""

FEW_SHOT_PROMPTS[("control_fixed", "python")] = FEW_SHOT_PROMPTS[("control", "python")]

FEW_SHOT_ANSWERS[("control", "python")] = """\
_ in range(t):
"""

FEW_SHOT_ANSWERS[("control_fixed", "python")] = FEW_SHOT_ANSWERS[("control", "python")]

FEW_SHOT_PROMPTS[("control", "java")] = """\
Complete the code in java to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        for (/* TODO: Your code here */) {
            int a = sc.nextInt();
            int b = sc.nextInt();
            int c = a + b;
            System.out.println(c);
        }
        sc.close();
    }
}
"""

FEW_SHOT_PROMPTS[("control_fixed", "java")] = FEW_SHOT_PROMPTS[("control", "java")]

FEW_SHOT_ANSWERS[("control", "java")] = """\
int i = 0; i < t; i++
"""

FEW_SHOT_ANSWERS[("control_fixed", "java")] = FEW_SHOT_ANSWERS[("control", "java")]

FEW_SHOT_PROMPTS[("control", "cpp")] = """\
Complete the code in cpp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

#include <iostream>
using namespace std;

int main() {
    int t, a, b, c;
    cin >> t;
    for (/* TODO: Your code here */) {
        cin >> a >> b;
        c = a + b;
        cout << c << endl;
    }
    return 0;
}
"""

FEW_SHOT_PROMPTS[("control_fixed", "cpp")] = FEW_SHOT_PROMPTS[("control", "cpp")]

FEW_SHOT_ANSWERS[("control", "cpp")] = """\
int i = 0; i < t; i++
"""

FEW_SHOT_ANSWERS[("control_fixed", "cpp")] = FEW_SHOT_ANSWERS[("control", "cpp")]

FEW_SHOT_PROMPTS[("control", "csharp")] = """\
Complete the code in csharp to solve this programming problem:

Description: You are given two integers $$$a$$$ and $$$b$$$. Print $$$a+b$$$.

Input Specification: The first line contains an integer $$$t$$$ ($$$1 \le t \le 10^4$$$) — the number of test cases in the input. Then $$$t$$$ test cases follow. Each test case is given as a line of two integers $$$a$$$ and $$$b$$$ ($$$-1000 \le a, b \le 1000$$$).

Output Specification: Print $$$t$$$ integers — the required numbers $$$a+b$$$.

Code:

using System;

class Program {
    static void Main() {
        int t = int.Parse(Console.ReadLine());
        for (/* TODO: Your code here */) {
            string[] inputs = Console.ReadLine().Split(' ');
            int a = int.Parse(inputs[0]);
            int b = int.Parse(inputs[1]);
            Console.WriteLine(c);
        }
    }
}
"""

FEW_SHOT_PROMPTS[("control_fixed", "csharp")] = FEW_SHOT_PROMPTS[("control", "csharp")]

FEW_SHOT_ANSWERS[("control", "csharp")] = """\
int i = 0; i < t; i++
"""

FEW_SHOT_ANSWERS[("control_fixed", "csharp")] = FEW_SHOT_ANSWERS[("control", "csharp")]

FEW_SHOT_PROMPTS[("api", "python")] = """\
Complete the code in python:

import numpy as np

n = 10
# Create an n by n array of ones
a = # TODO: Your code here
"""

FEW_SHOT_ANSWERS[("api", "python")] = """\
np.ones((n, n))
"""

FEW_SHOT_PROMPTS[("api", "java")] = """\
Complete the code in java:

import org.json.JSONObject;

public class Example {
    public static void main(String[] args) {
        String jsonData = "{\"name\":\"John\", \"age\":30}";
        JSONObject obj = /* TODO: Your code here */;

        String name = obj.getString("name");
        int age = obj.getInt("age");

        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
    }
}
"""

FEW_SHOT_ANSWERS[("api", "java")] = """\
new JSONObject(jsonData)
"""

FEW_SHOT_PROMPTS[("api", "cpp")] = """\
Complete the code in cpp:

#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // JSON data
    std::string jsonData = "{\"name\":\"John\", \"age\":30}";

    // Parse JSON data
    json j = /* TODO: Your code here */;

    // Extract data
    std::string name = j["name"];
    int age = j["age"];

    std::cout << "Name: " << name << std::endl;
    std::cout << "Age: " << age << std::endl;

    return 0;
}
"""

FEW_SHOT_ANSWERS[("api", "cpp")] = """\
json::parse(jsonData)
"""

FEW_SHOT_PROMPTS[("api", "csharp")] = """\
using System;
using Newtonsoft.Json.Linq;

class Program
{
    static void Main()
    {
        string jsonData = "{\"name\":\"John\", \"age\":30}";
        JObject jObject = /* TODO: Your code here */;

        string name = (string)jObject["name"];
        int age = (int)jObject["age"];

        Console.WriteLine("Name: " + name);
        Console.WriteLine("Age: " + age);
    }
}
"""

FEW_SHOT_ANSWERS[("api", "csharp")] = """\
JObject.Parse(jsonData)
"""

INSTRUCTION_TEMPLATE = jinja2.Template(
    'Replace the "{{placeholder}}" in the code above with the appropriate {{completion_type}}. Provide only the replaced {{completion_type}}.'
)

PREFIX_FEEDING_TEMPLATE = jinja2.Template(
    'Replace the "{{placeholder}}" in the code above with the appropriate {{completion_type}}.'
)

COMPLETION_TYPE_REPR = {
    "block": "block",
    "control": "control expression",
    "control_fixed": "control expression",
    "api": "API function call"
}


def get_infilling_parts(sample):
    parts = sample["prompt"].split(COMPLETION_PLACEHOLDER[sample["lang"]])
    assert len(parts) == 2
    return parts


def convert_to_l2r_prompt(sample):
    idx = sample["prompt"].find(COMPLETION_PLACEHOLDER[sample["lang"]])
    assert idx != -1
    return sample["prompt"][:idx]


def add_prefix_feeding(sample, completion_type):
    idx = sample["eval_prompt"].find("{{completion}}")
    prefix_feeding_instruction = PREFIX_FEEDING_TEMPLATE.render(
        placeholder=COMPLETION_PLACEHOLDER[sample["lang"]],
        completion_type=COMPLETION_TYPE_REPR[completion_type]
    )
    return (
        sample["prompt"] + "\n"
        + prefix_feeding_instruction + "\n\n"
        + sample["eval_prompt"][:idx]
    )


def add_instruct(sample, completion_type):
    instruction = INSTRUCTION_TEMPLATE.render(
        placeholder=COMPLETION_PLACEHOLDER[sample["lang"]],
        completion_type=COMPLETION_TYPE_REPR[completion_type]
    )
    return sample["prompt"] + "\n" + instruction + "\n\n"


def add_instruct_with_fewshot(sample, completion_type):
    instruction = INSTRUCTION_TEMPLATE.render(
        placeholder=COMPLETION_PLACEHOLDER[sample["lang"]],
        completion_type=COMPLETION_TYPE_REPR[completion_type]
    )
    return (
        FEW_SHOT_PROMPTS[(completion_type, sample["lang"])] + "\n"
        + instruction + "\n\n"
        + FEW_SHOT_ANSWERS[(completion_type, sample["lang"])] + "\n\n"
        + add_instruct(sample, completion_type)
    )


def extract_code_from_chat_model(code, lang, completion_type):
    # Extract all code blocks
    in_code = False
    res = []
    res_inline = []
    cur_code_block = []
    for line in code.splitlines(keepends=True):
        if line.startswith("```"):
            if in_code:
                in_code = False
                res.append("".join(cur_code_block).rstrip())
                cur_code_block = []
            else:
                in_code = True
        else:
            if in_code:
                cur_code_block.append(line)
            else:
                if "`" in line:
                    start = line.find('`') + 1
                    end = line.find('`', start)
                    if end != -1:
                        res_inline.append(line[start:end])

    if not res:
        if not res_inline:
            return code.rstrip()
        else:
            res = res_inline

    # Resolve conflicts if there are multiple
    new_res = []
    for i, code in enumerate(res):
        if COMPLETION_PLACEHOLDER[lang].strip() in code:
            new_res.append((-2, code))
        elif FEW_SHOT_ANSWERS[(completion_type, lang)].strip() in code:
            new_res.append((-1, code))
        else:
            new_res.append((len(res) - i, code))
    return list(sorted(new_res, reverse=True))[0][1]


def truncate_for_fewshot(code):
    idx = code.find("\nComplete the code in ")
    if idx != -1:
        code = code[:idx]
        if code.endswith("\n"):
            code = code[:-1]
    return code


def truncate_line_until_parsable(sample, code):
    parser = get_parser(sample["lang"])
    lines = code.splitlines(keepends=True)
    while lines:
        full_code = sample["eval_prompt"].replace("{{completion}}", "".join(lines))
        code_bytes = full_code.encode("utf-8")
        tree = parser.parse(code_bytes)
        error_check = ErrorCheckVisitor()
        error_check(tree)
        if error_check.error_cnt > 0:
            lines.pop()
        else:
            break
    return "".join(lines)


def match_prefix_and_suffix(l1, l2):
    p = 0
    while p < len(l1) and p < len(l2):
        if l1[p] == l2[p]:
            p += 1
        else:
            break
    q = 0
    while -q < len(l1) and -q < len(l2):
        if l1[q - 1] == l2[q - 1]:
            q -= 1
        else:
            break
    return p, q


def truncate_line_until_block(sample, code):
    parser = get_parser(sample["lang"])
    lines = code.splitlines(keepends=True)
    while lines:
        eval_prefix, eval_suffix = sample['eval_prompt'].split("{{completion}}")
        eval_prefix = eval_prefix.encode("utf-8")
        eval_suffix = eval_suffix.encode("utf-8")
        completion = "".join(lines).encode("utf-8")
        if sample["lang"] == "python":
            code_bytes_0 = eval_prefix + b"pass" + eval_suffix
        else:
            code_bytes_0 = eval_prefix + eval_suffix
        code_bytes_1 = eval_prefix + completion + eval_suffix

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_1)
        visitor(tree)
        if visitor.error_cnt > 0:
            lines.pop()
            continue
        visitor_trace_1 = [(x, y) for _, x, y in visitor.ndtypes]

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_0)
        visitor(tree)
        assert visitor.error_cnt == 0
        visitor_trace_0 = [(x, y) for _, x, y in visitor.ndtypes]
        if len(visitor_trace_0) > len(visitor_trace_1):
            lines.pop()
            continue

        prefix_matched, suffix_matched = match_prefix_and_suffix(visitor_trace_0, visitor_trace_1)
        matched_diff = len(visitor_trace_0) - (prefix_matched - suffix_matched)
        if sample["lang"] == "python":
            matched_diff -= 4
        if matched_diff == 0:
            break
        else:
            lines.pop()
    return "".join(lines)


def truncate_control(sample, completion, remove_colon=False):
    if sample["lang"] == "python":
        completion = truncate_to_first_line(completion)
        if remove_colon:
            completion = completion.rstrip()
            if completion.endswith(":"):
                completion = completion[:-1]
        return completion
    else:
        depth = 0
        for i, ch in enumerate(completion):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == -1:
                return completion[:i]
        return completion


def truncate_api_call(completion):
    depth = 0
    for i, ch in enumerate(completion):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth <= 0:
                return completion[:i + 1]
    return completion


def truncate_stop_words(completion):
    stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]
    min_stop_index = len(completion)
    for stop_word in stop_words:
        stop_index = completion.find(stop_word)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return completion[:min_stop_index]


def apply_prompt(
    sample: dict,
    completion_type: str,
    mode: str,
    model_wrapper: ModelWrapper,
) -> str:
    if mode == "infilling":
        prefix, suffix = get_infilling_parts(sample)
        prompt = model_wrapper.assemble_infilling_prompt(prefix, suffix, reverse=False)
    elif mode == "reverse_infilling":
        prefix, suffix = get_infilling_parts(sample)
        prompt = model_wrapper.assemble_infilling_prompt(prefix, suffix, reverse=True)
    elif mode == "left_to_right":
        prompt = convert_to_l2r_prompt(sample)
    elif mode == "prefix_feeding":
        prompt = add_prefix_feeding(sample, completion_type)
    elif mode == "instructed":
        prompt = add_instruct(sample, completion_type)
    elif mode == "fewshot":
        prompt = add_instruct_with_fewshot(sample, completion_type)
    elif mode == "raw":
        prompt = sample["prompt"]
    else:
        raise ValueError(mode)
    return prompt


def apply_postprocessors(
    completion: str,
    sample: dict,
    completion_type: str,
    post_processors: Iterable[str]
) -> str:
    if post_processors is None:
        post_processors = []
    for post_processor in post_processors:
        if post_processor == "extract_code":
            completion = extract_code_from_chat_model(completion, sample["lang"], completion_type)
        elif post_processor == "truncate_line":
            completion = truncate_to_first_line(completion)
        elif post_processor == "truncate_line_lf":
            completion = truncate_to_first_line(completion, add_line_break=True)
        elif post_processor == "truncate_fewshot":
            completion = truncate_for_fewshot(completion)
        elif post_processor == "truncate_line_until_parsable":
            completion = truncate_line_until_parsable(sample, completion)
        elif post_processor == "truncate_line_until_block":
            completion = truncate_line_until_block(sample, completion)
        elif post_processor == "truncate_control":
            completion = truncate_control(sample, completion)
        elif post_processor == "truncate_control_remove_colon":
            completion = truncate_control(sample, completion, remove_colon=True)
        elif post_processor == "truncate_api_call":
            completion = truncate_api_call(completion)
        elif post_processor == "truncate_stop_words":
            completion = truncate_stop_words(completion)
        else:
            raise ValueError(post_processor)
    return completion
