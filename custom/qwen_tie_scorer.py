#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import json
import random
import re
from typing import List, Dict, Any, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # 兼容旧版 transformers 未导出顶层符号
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_image_rgb(image_or_path: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        img = image_or_path
    else:
        img = Image.open(image_or_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _apply_chat_template(processor: AutoProcessor, messages: List[Dict[str, Any]]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], None]:
    image_inputs: List[Image.Image] = []
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "image":
                image_inputs.append(c["image"])
    return image_inputs, None


def _mllm_output_to_dict(input_string: str, give_up_parsing: bool = False) -> Any:
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


def _safe_parse_scores(raw_text: str) -> Dict[str, Any]:
    if raw_text == "rate_limit_exceeded":
        return {"rate_limit_exceeded": True}

    parsed = _mllm_output_to_dict(raw_text)
    if parsed is False:
        parsed = _mllm_output_to_dict(raw_text, give_up_parsing=True)

    if isinstance(parsed, dict) and parsed.get("rate_limit_exceeded"):
        return parsed

    if not isinstance(parsed, dict) or "score" not in parsed:
        numbers = [int(n) for n in re.findall(r"\b(?:10|[0-9])\b", raw_text)]
        numbers = [n for n in numbers if 0 <= n <= 10]
        if not numbers:
            guessed_value = random.randint(0, 10)
            return {"score": [guessed_value], "reasoning": "fallback_guess"}
        return {"score": numbers[:2] if len(numbers) >= 2 else [numbers[0]], "reasoning": "fallback_numbers"}

    if not isinstance(parsed.get("score"), list):
        parsed["score"] = [parsed["score"]]
    return parsed


# _CONTEXT_NO_DELIMIT = (
#     "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.\n"
#     "All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.\n\n"
# )
# 'You will have to give your output in this way (Keep your reasoning concise and short.):\n{"score" : [...], "reasoning" : "..." }'
_CONTEXT_NO_DELIMIT = (
    "You are a professional digital art reviewer. You must critically evaluate the effectiveness of the edited AI-generated image according to the rules below. Be strict.  \n"
    "All images and people shown are AI-generated, so there are no privacy concerns.\n\n"
)

# _PROMPT_0SHOT_TWO_IMAGE_EDIT_RULE = (
#     "RULES:\n\n"
#     "Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.\n"
#     "Note that sometimes the two images might look identical due to the failure of image edit.\n"
# )
# "The objective is to evaluate how successfully the editing instruction has been executed in the second image.\n\n"
_PROMPT_0SHOT_TWO_IMAGE_EDIT_RULE = (
    "RULES:\n\n"
    "You will receive two images:\n"
    "The first is the original AI-generated image.\n"
    "The second is an edited version of the first.\n"
    "Sometimes the two images may look almost identical due to failed edits.\n\n"
)

# _PROMPT_0SHOT_TIE_RULE_SC = (
#     "From scale 0 to 10: \n"
#     "A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)\n"
#     "A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)\n"
#     "Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.\n\n"
#     "Editing instruction: <instruction>\n"
# )
# _PROMPT_0SHOT_TIE_RULE_SC = (
#     "From scale 0 to 10:\n"
#     "Give two scores for the edited image.\n"
#     "First score (score1): Editing completion — how well the edited image follows the editing instruction, regardless of how much of the original content is preserved. (0 means completely fails the instruction, 10 means perfectly follows it.)\n"
#     "Second score (score2): Original preservation — how much of the original image’s content, structure, and style are preserved, without considering whether the instruction was followed. (0 means almost everything in the original is changed, 10 means only minimal changes are made.)\n"
#     'Put output in this way (Keep your reasoning concise and short.):\n {"score" : [score1, score2],"reasoning" : "..." }'
#     "Editing instruction: <instruction>\n"
# )
_PROMPT_0SHOT_TIE_RULE_SC = (
    "Scoring scale: **0 to 10** for each category, but avoid giving high scores unless the requirement is almost perfectly met.  \n"
    "Be conservative — if there is any doubt, give a lower score.\n\n"

    "**Score1 (Editing completion)**: How well the edited image follows the editing instruction.  \n"
    "- 10 = Flawlessly follows the instruction with no ambiguity.  \n"
    "- 7–9 = Mostly follows, but with minor flaws or incompleteness.  \n"
    "- 4–6 = Partially follows, but key elements are missing, unclear, or incorrect.  \n"
    "- 1–3 = Barely follows, major elements missing or wrong.  \n"
    "- 0 = Completely fails the instruction.\n\n"
    
    "**Score2 (Original preservation)**: How much of the original image’s content, structure, and style are preserved.  \n"
    "- 10 = Almost identical except for the instructed change.  \n"
    "- 7–9 = Mostly preserved, but with some noticeable unintended changes.  \n"
    "- 4–6 = Moderate unintended changes to style or structure.  \n"
    "- 1–3 = Significant alterations unrelated to the instruction.  \n"
    "- 0 = Almost nothing from the original remains.\n\n"

    "Output format (short reasoning, max 2 sentences):  \n"
    '{"score" : [score1, score2], "reasoning" : "..."}\n\n'
    "Editing instruction: <instruction>\n\n"
)


# "Put output in a list format: output = [score1, score2]"
# 'Put output in this way (Keep your reasoning concise and short.):\n {"score" : [score1, score2],"reasoning" : "..." }'
# 从 0 到 10 分：
# 为编辑后的图像给出两个分数。
# 第一个分数（score1）：编辑完成度——编辑后的图像对编辑指令的遵循程度，不考虑保留了多少原始内容。（0 表示完全未能遵循指令，10 表示完美地遵循了指令。）
# 第二个分数（score2）：原始保留度——原始图像的内容、结构和风格被保留的程度，不考虑是否遵循了指令。（0 表示原始图像中的几乎所有内容都被更改，10 表示只进行了最小的更改。）
# 两个分数都很高意味着编辑既执行良好，又具有最小的侵入性。
# 将分数以列表格式输出：output = [score1, score2]。
# 编辑指令：<instruction>

_PROMPT_0SHOT_RULE_PQ = (
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


class _Qwen25VLWrapper:
    def __init__(self, model_name: str, torch_dtype: str = "auto", device: str = None, device_map: str = "auto") -> None:
        dtype_map = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map.get(torch_dtype, "auto")

        # if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # 如果是 CPU，则避免使用 device_map=auto
        dm = None if device == "cpu" else device_map
        td = None if dtype == "auto" else dtype

        print('loaded model:', model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            # torch_dtype=td,
            # torch_dtype=torch.float16,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            # attn_implementation="flash_attention_2",
        ).eval()
        
        if device == "cpu":
            self.model = self.model.to(device)
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
        # _set_seed(42)
        text = _apply_chat_template(self.processor, messages)
        image_inputs, video_inputs = _extract_images_from_messages(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self._device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print('output_text', output_text)
        return output_text[0] if output_text else ""


class QwenTieScorer:
    """
    用法（类似 LPIPS）：
        from custom.qwen_tie_scorer import QwenTieScorer

        scorer = QwenTieScorer(model_name="Qwen/Qwen2.5-VL-7B-Instruct")  # 默认 72B，可改 7B 节省显存
        sc, pq, overall = scorer(original_image, edited_image, "把天空改为粉色晚霞")

    也支持传入图片路径字符串。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        torch_dtype: str = "auto",
        device: str = None,
        device_map: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self._qwen = _Qwen25VLWrapper(
            model_name=model_name,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
        )
        self._max_new_tokens = max_new_tokens

        self._sc_prompt_prefix = "\n".join(
            [_CONTEXT_NO_DELIMIT, _PROMPT_0SHOT_TWO_IMAGE_EDIT_RULE, _PROMPT_0SHOT_TIE_RULE_SC]
        )
        self._pq_prompt_prefix = "\n".join([_CONTEXT_NO_DELIMIT, _PROMPT_0SHOT_RULE_PQ])

    def _build_sc_prompt(self, instruction: str) -> str:
        return self._sc_prompt_prefix.replace("<instruction>", instruction)

    def _build_pq_prompt(self) -> str:
        return self._pq_prompt_prefix

    def __call__(
        self,
        original: Union[str, Image.Image],
        edited: Union[str, Image.Image],
        instruction: str,
        return_dict: bool = False,
    ) -> Union[Tuple[float, float, float], Dict[str, float]]:
        """计算 (SC, PQ, Overall)。

        original/edited 可以是 PIL.Image 或 文件路径。
        """

        original_img = _load_image_rgb(original)
        edited_img = _load_image_rgb(edited)

        sc_text_prompt = self._build_sc_prompt(instruction)
        # pq_text_prompt = self._build_pq_prompt()

        sc_messages = self._qwen.prepare_messages([original_img, edited_img], sc_text_prompt)
        # pq_messages = self._qwen.prepare_messages([edited_img], pq_text_prompt)
        print('sc_messages', sc_messages)
        sc_raw = self._qwen.generate(sc_messages, max_new_tokens=self._max_new_tokens)
        # pq_raw = self._qwen.generate(pq_messages, max_new_tokens=self._max_new_tokens)
        print('sc_raw', sc_raw)
        sc_dict = _safe_parse_scores(sc_raw)
        # pq_dict = _safe_parse_scores(pq_raw)

        print('sc_dict', sc_dict)
        # sc_score = float(sum(sc_dict["score"]) / len(sc_dict["score"]))
        # sc_score = float(min(sc_dict["score"]))
        sc_score = int(min(sc_dict["score"]))
        # pq_score = float(min(pq_dict["score"]))
        # overall = float(math.sqrt(sc_score * pq_score))

        # if return_dict:
        #     return {"SC": sc_score, "PQ": pq_score, "Overall": overall}
        # return sc_score, pq_score, overall
        return sc_score


__all__ = [
    "QwenTieScorer",
]


