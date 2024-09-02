import os
import time
from argparse import Namespace

import openai
import tiktoken
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, NoBadWordsLogitsProcessor


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


class CodeLlamaFromFile(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        config = transformers.LlamaConfig.from_pretrained(model_name, from_safetensors=True)
        device = torch.device("cuda")
        self.model = transformers.LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
        ).to(device)
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
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        if "<EOT>" in generated_text:
            generated_text = generated_text[:generated_text.index("<EOT>")]
        return generated_text

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


class PhiModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("microsoft/phi")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        device = torch.device("cuda")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 50256
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
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
            prompt, truncation=True, return_attention_mask=False, return_tensors="pt"
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


class MixtralModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("mistralai/Mixtral")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['▁#', '▁/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=True, return_attention_mask=False, return_tensors="pt"
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


class WizardModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("WizardLM/WizardCoder")
        if model_name == "WizardLM/WizardCoder-33B-V1.1":
            self.use_deepseek_base = True
        else:
            self.use_deepseek_base = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
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
            prompt, truncation=True, return_attention_mask=False, return_tensors="pt"
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
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if self.use_deepseek_base:
            if reverse:
                return "<｜fim▁begin｜>" + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>" + prefix
            else:
                return "<｜fim▁begin｜>" + prefix + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>"
        else:
            if reverse:
                return "<fim_prefix>" + "<fim_suffix>" + suffix + "<fim_middle>" + prefix
            else:
                return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"


class SantacoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("bigcode/santacoder")
        if ":" in model_name:
            model_name, revision = model_name.split(":")
        else:
            revision = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32
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
                pad_token_id=self.tokenizer.unk_token_id,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<fim-prefix>" + "<fim-suffix>" + suffix + "<fim-middle>" + prefix
        else:
            return "<fim-prefix>" + prefix + "<fim-suffix>" + suffix + "<fim-middle>"


class MagicCoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("ise-uiuc/Magicoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.pipeline = transformers.pipeline(
            model=model_name,
            tokenizer=self.tokenizer,
            task="text-generation",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
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
        generated_text = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.2,
            max_new_tokens=128,
            top_p=0.95,
            handle_long_generation="hole",
            logits_processor=self.logits_processor
        )[0]["generated_text"]
        return generated_text[len(prompt):]

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


def build_model(args: Namespace) -> ModelWrapper:
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
    elif args.model_name.startswith("microsoft/phi"):
        model_wrapper = PhiModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("mistralai/Mixtral"):
        model_wrapper = MixtralModel(args.model_name, 4096, args.block_comments)
    elif args.model_name.startswith("WizardLM/WizardCoder"):
        model_wrapper = WizardModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("bigcode/santacoder"):
        model_wrapper = SantacoderModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("ise-uiuc/Magicoder"):
        model_wrapper = MagicCoderModel(args.model_name, 4096, args.block_comments)
    else:
        model_wrapper = CodeLlamaFromFile(args.model_name, 4096, args.block_comments)
    return model_wrapper
