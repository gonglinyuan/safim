import argparse
import gzip
import json
import os
import time
from typing import Dict, Iterable

import jinja2
import openai
import tiktoken
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, NoBadWordsLogitsProcessor
from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    'build/tree_sitter.so',

    # Include one or more languages
    [
        'tree-sitter-python',
        'tree-sitter-java',
        'tree-sitter-cpp',
        'tree-sitter-c-sharp'
    ]
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

    def __init__(self, with_ndtypes=False, print_debug_outputs=False):
        self.with_ndtypes = with_ndtypes
        self.print_debug_outputs = print_debug_outputs
        self.stack = []
        self.ndtypes = []

    def enter(self, node) -> bool:
        return True

    def leave(self, node):
        pass

    def enter_leaf(self, node):
        pass

    def print_stack(self, node):
        depth = len(self.stack)
        print(" " * depth * 2 + node.type)

    def on_enter(self, node) -> bool:
        if self.print_debug_outputs:
            self.print_stack(node)
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
    def __init__(self, with_ndtypes=False):
        super().__init__(with_ndtypes)
        self.error_cnt = 0

    def enter_ERROR(self, node):
        if node.text.decode("utf-8") != ";":
            self.error_cnt += 1


class ModelWrapper:
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError()

    def invoke_cached(self, prompt: str, cache: dict) -> str:
        if prompt not in cache:
            cache[prompt] = self.invoke(prompt)
        return cache[prompt]

    def assemble_completion_prompt(self, prompt: str) -> str:
        return prompt

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        raise NotImplementedError()


class CodeLlama(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("codellama/CodeLlama")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(["#", "▁#", "/*", "▁/*"])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        generated_text = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.2,
            max_new_tokens=128,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=0.95,
            handle_long_generation="hole",
            logits_processor=self.logits_processor
        )[0]["generated_text"]
        return generated_text[len(prompt):]

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<PRE>" + " <SUF>" + suffix + " <MID>" + prefix
        else:
            return "<PRE>" + prefix + " <SUF>" + suffix + " <MID>"


class Incoder(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("facebook/incoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        device = torch.device("cuda")
        if model_name.endswith("6B"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(
                        ['#Ġ', '/*Ġ', 'Ġ#Ġ', 'ĠĠ#Ġ', 'ĠĠ/*Ġ', 'ĠĠĠ#Ġ', 'ĠĠĠ/*Ġ', 'ĠĠĠĠ#Ġ', 'ĠĠĠĠ/*Ġ', 'ĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠ/*Ġ', 'ĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠ/*Ġ', 'ĠĠĠĠĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ']
                    )],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_length=min(input_ids_len + 128, self.max_length),
                top_p=0.95,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:], clean_up_tokenization_spaces=False
        )
        EOM = "<|endofmask|>"
        if EOM not in generated_text:
            generated_text += EOM
        return generated_text[:generated_text.index(EOM)]

    @staticmethod
    def make_sentinel(i):
        # signals (1) a location to insert an infill and (2) the start of the infill generation
        return f"<|mask:{i}|>"

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return prefix + self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0)
        else:
            return self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0) + prefix


class OpenAIModel(ModelWrapper):
    def __init__(self, model_name, max_length):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def invoke(self, prompt: str) -> str:
        cnt = 0
        tokenized = self.tokenizer.encode_ordinary(prompt)
        if len(tokenized) > self.max_length:
            tokenized = tokenized[-self.max_length:]
            prompt = self.tokenizer.decode(tokenized)
        while True:
            if cnt == 999:
                return ""
            try:
                result = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
                break
            except Exception as e:
                if "Please try again with a different prompt" in str(e):
                    return ""
                cnt += 1
                time.sleep(5)
                print(f"{e}")
        return result["choices"][0]["message"]["content"]

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class CodegenModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("Salesforce/codegen")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        device = torch.device("cuda")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 50256
        if "6B" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_length=min(input_ids_len + 128, self.max_length),
                pad_token_id=50256,
                top_p=0.95,
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class StarcoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("bigcode/starcoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_length=min(input_ids_len + 128, self.max_length),
                top_p=0.95,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<fim_prefix>" + "<fim_suffix>" + suffix + "<fim_middle>" + prefix
        else:
            return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"


class DeepseekModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("deepseek-ai/deepseek")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=1,
                    temperature=0.2,
                    max_length=min(input_ids_len + 128, self.max_length),
                    top_p=0.95,
                    logits_processor=self.logits_processor
                )
            except RuntimeError as e:
                print(e)
                return ""
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<｜fim▁begin｜>" + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>" + prefix
        else:
            return "<｜fim▁begin｜>" + prefix + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>"


def truncate_to_first_line(code):
    lines = code.splitlines()
    for line in lines:
        if line.strip():
            return line
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

FEW_SHOT_ANSWERS[("control", "python")] = """\
_ in range(t):
"""

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

FEW_SHOT_ANSWERS[("control", "java")] = """\
int i = 0; i < t; i++
"""

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

FEW_SHOT_ANSWERS[("control", "cpp")] = """\
int i = 0; i < t; i++
"""

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

FEW_SHOT_ANSWERS[("control", "csharp")] = """\
int i = 0; i < t; i++
"""

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
    parser = Parser()
    parser.set_language(TS_LANG[sample["lang"]])
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
    parser = Parser()
    parser.set_language(TS_LANG[sample["lang"]])
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


def truncate_control(sample, completion):
    if sample["lang"] == "python":
        return truncate_to_first_line(completion)
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


def handle_example(
    sample: dict,
    completion_type: str,
    mode: str,
    model_wrapper: ModelWrapper,
    cache: dict,
    post_processors: Iterable[str]
):
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
    else:
        raise ValueError(mode)
    completion = model_wrapper.invoke_cached(prompt, cache)
    if post_processors is None:
        post_processors = []
    for post_processor in post_processors:
        if post_processor == "extract_code":
            completion = extract_code_from_chat_model(completion, sample["lang"], completion_type)
        elif post_processor == "truncate_line":
            completion = truncate_to_first_line(completion)
        elif post_processor == "truncate_fewshot":
            completion = truncate_for_fewshot(completion)
        elif post_processor == "truncate_line_until_parsable":
            completion = truncate_line_until_parsable(sample, completion)
        elif post_processor == "truncate_line_until_block":
            completion = truncate_line_until_block(sample, completion)
        elif post_processor == "truncate_control":
            completion = truncate_control(sample, completion)
        elif post_processor == "truncate_api_call":
            completion = truncate_api_call(completion)
        else:
            raise ValueError(post_processor)
    return {
        "task_id": sample["task_id"],
        "completion": completion
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("completion_type", type=str)
    parser.add_argument("cache_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("--block_comments", action="store_true")
    parser.add_argument("--post_processors", type=str, nargs="+")

    args = parser.parse_args()

    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    if args.model_name.startswith("codellama/CodeLlama"):
        model_wrapper = CodeLlama(args.model_name, 4096, args.block_comments)
    elif args.model_name.startswith("facebook/incoder"):
        model_wrapper = Incoder(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("gpt-"):
        model_wrapper = OpenAIModel(args.model_name, 4096 - 32)
    elif args.model_name.startswith("Salesforce/codegen"):
        model_wrapper = CodegenModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("bigcode/starcoder"):
        model_wrapper = StarcoderModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("deepseek-ai/deepseek"):
        model_wrapper = DeepseekModel(args.model_name, 4096, args.block_comments)
    else:
        raise ValueError(args.model_name)

    outputs = []
    try:
        for sample in tqdm(stream_jsonl(args.data_path)):
            outputs.append(
                handle_example(sample, args.completion_type, args.mode, model_wrapper, cache, args.post_processors)
            )
            if len(outputs) == 10:
                for o in outputs:
                    print("=====", o["task_id"], "=====")
                    print(o["completion"])
                    print("====================")
                    print()
            if len(outputs) % 1000 == 0:
                with open(args.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
        success = True
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        success = False
    with open(args.cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    if success:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()
