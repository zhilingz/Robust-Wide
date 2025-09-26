#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import json
import random
import re
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_rgb(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def apply_chat_template(processor: AutoProcessor, messages: List[Dict[str, Any]]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], None]:
    image_inputs: List[Image.Image] = []
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "image":
                image_inputs.append(c["image"])
    return image_inputs, None


def mllm_output_to_dict(input_string: str, give_up_parsing: bool = False) -> Any:
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    delimiter = "||V^=^V||"

    def verify(s: str, target_sequence: str) -> bool:
        return s.count(target_sequence) == 2

    def is_int_between_0_and_10(s: str) -> bool:
        try:
            v = int(s)
            return 0 <= v <= 10
        except Exception:
            return False

    def fix_json(input_str: str) -> str:
        fixed_str = re.sub(r'(\w+):', r'"\1":', input_str)

        def format_value(match):
            key, value, comma = match.groups()
            value = value.strip()
            if re.match(r'^-?\d+(\.\d+)?$', value):
                value = f'[{value}]'
            elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
                pass
            else:
                value = f'"{value}"'
            return f'{key}: {value}{comma}'

        fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)
        return fixed_str

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            return False
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        start_index = input_string.find("{")
        end_index = input_string.rfind("}") + 1
        if start_index == -1 or end_index == 0:
            start_index = input_string.find("[")
            end_index = input_string.rfind("]") + 1
            if give_up_parsing:
                guessed_value = random.randint(0, 10)
                json_content = {
                    "score": [guessed_value],
                    "reasoning": f"guess_if_cannot_parse | {input_string}",
                }
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r"^\[\d+(?:\s*,\s*\d+)*\]$", input_string[start_index:end_index] or ""):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {"score": scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string.strip()):
                scores = [int(input_string.strip())]
                json_content = {"score": scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                return False

    if start_index != -1 and end_index != -1 and start_index != end_index:
        json_str = input_string[start_index:end_index].strip().replace("\n", "")
        try:
            new_data = json.loads(json_str)
            if not isinstance(new_data.get("score"), list):
                new_data["score"] = [new_data["score"]]
        except Exception:
            try:
                new_data = json.loads(fix_json(json_str))
            except Exception:
                return False
        return new_data
    return False


CONTEXT_NO_DELIMIT = (
    "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.\n"
    "All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.\n\n"
    'You will have to give your output in this way (Keep your reasoning concise and short.):\n{\n"score" : [...],\n"reasoning" : "..." \n}'
)

PROMPT_0SHOT_TWO_IMAGE_EDIT_RULE = (
    "RULES:\n\n"
    "Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.\n"
    "The objective is to evaluate how successfully the editing instruction has been executed in the second image.\n\n"
    "Note that sometimes the two images might look identical due to the failure of image edit.\n"
)

PROMPT_0SHOT_TIE_RULE_SC = (
    "From scale 0 to 10: \n"
    "A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)\n"
    "A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)\n"
    "Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.\n\n"
    "Editing instruction: <instruction>\n"
)

PROMPT_0SHOT_RULE_PQ = (
    "RULES:\n\n"
    "The image is an AI-generated image.\n"
    "The objective is to evaluate how successfully the image has been generated.\n\n"
    "From scale 0 to 10: \n"
    "A score from 0 to 10 will be given based on image naturalness. \n"
    "( \n"
    "    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. \n"
    "    10 indicates that the image looks natural.\n"
    ")\n"
    "A second score from 0 to 10 will rate the image artifacts. \n"
    "( \n"
    "    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. \n"
    "    10 indicates the image has no artifacts.\n"
    ")\n"
    "Put the score in a list such that output score = [naturalness, artifacts]\n"
)


class Qwen25VLWrapper:
    def __init__(self, model_name: str, torch_dtype: str = "auto", device_map: str = "auto") -> None:
        dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}.get(torch_dtype, "auto")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype if dtype != "auto" else None,
            device_map=device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def prepare_messages(self, images: List[Image.Image], text_prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text_prompt}],
            }
        ]

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512) -> str:
        set_seed(42)
        text = apply_chat_template(self.processor, messages)
        image_inputs, video_inputs = extract_images_from_messages(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if device == "cpu":
            self.model = self.model.to(device)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""


class TieScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ") -> None:
        self.qwen = Qwen25VLWrapper(model_name=model_name)

        self.sc_prompt_prefix = "\n".join(
            [CONTEXT_NO_DELIMIT, PROMPT_0SHOT_TWO_IMAGE_EDIT_RULE, PROMPT_0SHOT_TIE_RULE_SC]
        )
        self.pq_prompt_prefix = "\n".join([CONTEXT_NO_DELIMIT, PROMPT_0SHOT_RULE_PQ])

    def _build_sc_prompt(self, instruction: str) -> str:
        return self.sc_prompt_prefix.replace("<instruction>", instruction)

    def _build_pq_prompt(self) -> str:
        return self.pq_prompt_prefix

    def score(self, original: Image.Image, edited: Image.Image, instruction: str) -> Tuple[float, float, float]:
        sc_text_prompt = self._build_sc_prompt(instruction)
        pq_text_prompt = self._build_pq_prompt()

        sc_messages = self.qwen.prepare_messages([original, edited], sc_text_prompt)
        pq_messages = self.qwen.prepare_messages([edited], pq_text_prompt)

        sc_raw = self.qwen.generate(sc_messages)
        pq_raw = self.qwen.generate(pq_messages)

        sc_dict = mllm_output_to_dict(sc_raw, give_up_parsing=True)
        pq_dict = mllm_output_to_dict(pq_raw, give_up_parsing=True)
        if sc_dict == "rate_limit_exceeded" or pq_dict == "rate_limit_exceeded":
            raise RuntimeError("rate_limit_exceeded")

        sc_score = min(sc_dict["score"])
        pq_score = min(pq_dict["score"])
        overall = math.sqrt(sc_score * pq_score)
        return float(sc_score), float(pq_score), float(overall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True, help="原图路径")
    parser.add_argument("--edited", type=str, required=True, help="编辑后图路径")
    parser.add_argument("--instruction", type=str, required=True, help="编辑指令")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        help="Hugging Face 模型名称（可改成 Qwen/Qwen2.5-VL-7B-Instruct 以降低显存）",
    )
    parser.add_argument("--output", type=str, default="", help="可选，将评分结果保存为 JSON 文件路径")
    args = parser.parse_args()

    original = load_image_rgb(args.original)
    edited = load_image_rgb(args.edited)

    scorer = TieScorer(model_name=args.model)
    sc, pq, overall = scorer.score(original, edited, args.instruction)

    result = {"SC": sc, "PQ": pq, "Overall": overall}
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()