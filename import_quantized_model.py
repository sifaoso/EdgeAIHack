from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import requests
from PIL import Image

model_id = "matChmp/llava-phi-3-mini-hf-4-bit-2"
prompt = "<|user|>\n<image>\nIm a blind person, can you describe me the scene with high precision so i can walk safely? Start your answer directly with the description. <|end|>\n<|assistant|>\n"
image_file = "journal.jpeg"


model = LlavaForConditionalGeneration.from_pretrained(
    model_id,  
    low_cpu_mem_usage=True, 
    device_map = "auto",
)
model.tie_weights()

processor = AutoProcessor.from_pretrained(model_id)
processor.patch_size = model.config.vision_config.patch_size
processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

raw_image = Image.open(image_file)
inputs = processor(raw_image, prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
