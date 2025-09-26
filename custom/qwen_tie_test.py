import sys
sys.path.append('/public/zhangzhiling/code/Robust-Wide')
sys.path.append('/public/zhangzhiling/code/custom')

from custom.qwen_tie_scorer import QwenTieScorer
from PIL import Image

# 1) 初始化（建议用7B版本更省显存）
scorer = QwenTieScorer(model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
# scorer = QwenTieScorer(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

# 2) 传路径
sc = scorer(
    "train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step20000/image.png", 
    "train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step20000/generated_image_before_wm.png", 
    "turn the scene into a snowy winter wonderland"
    # "inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png",
    # "inference_results/llm_test_instructpix2pix_10_1.0_10.0.png",
    # "inference_results/llm_test_MagicBrush_3_1.0_1.png",
    # "inference_results/llm_test_MagicBrush_3_1.0_1.5.png",
    # "inference_results/llm_test_MagicBrush_3_1.0_2.5.png",
    # "put her in a windmill"
    # "train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step16000/image.png",
    # "train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step16000/generated_image_before_wm.png",
    # "Transform the train tracks into a river"
    )
print('sc_score', sc)  

# 3) 或传 PIL.Image
# orig = Image.open("train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step20000/image.png").convert("RGB")
# edit = Image.open("train_results/2025-08-12T05-13-41_BleachNick___ultra_edit_500k_magicbrush-jul7_257754/step20000/generated_image_before_wm.png").convert("RGB")
# result = scorer(orig, edit, "turn the scene into a snowy winter wonderland", return_dict=True)  # {'SC':..., 'PQ':..., 'Overall':...}
# print('2', result)