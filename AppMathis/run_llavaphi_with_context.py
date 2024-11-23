# from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor
# import torch
# import requests
# from PIL import Image
# from transformers import BitsAndBytesConfig

# model_id = "matChmp/llava-phi-3-mini-hf-4-bit-2"
# prompt = "<|user|>\n<image>\nIm a blind person, can you describe me the scene with high precision so i can walk safely? Start your answer directly with the>"
# image_file = "journal.jpeg"


# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     low_cpu_mem_usage=True,
#     device_map = "auto",
# )
# model.tie_weights()

# processor = AutoProcessor.from_pretrained(model_id)
# processor.patch_size = model.config.vision_config.patch_size
# processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

# raw_image = Image.open(image_file)
# inputs = processor(raw_image, prompt, return_tensors='pt').to(0, torch.float16)

# output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))

import argparse
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# Function to process the image and generate a description
def describe_image(image_path, model_id, prompt):
    # Load the model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.tie_weights()

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    # Load the image
    raw_image = Image.open(image_path)

    # Prepare inputs
    inputs = processor(raw_image, prompt, return_tensors="pt").to(0, torch.float16)

    # Generate output
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    #response = output.split("<|assistant|>")[-1].strip()
    # Decode and return the description
    return processor.decode(output[0], skip_special_tokens=True)
    #return response
# Main function to parse arguments
def main():
    print("hello")
    parser = argparse.ArgumentParser(description="Describe an image using the Llava model.")
    parser.add_argument(
        "image_path", type=str, help="Path to the image file to be described."
    )

    parser.add_argument(
        "contexte",  # Nom de l'argument (avec "--" pour indiquer un argument optionnel)
        type=str,  # Type attendu : chaîne de caractères
    )

    args = parser.parse_args()

    # Constants
    model_id = "matChmp/llava-phi-3-mini-hf-4-bit-2"
    prompt = (
        f"<|user|>\n<image>\nIm a blind person, can you describe me the scene with high precision so I can walk safely? For the context, the person on the scene is <|context|>{args.contexte}. Use the <|context|> to describe the scene. <|end|>\n<|assistant|>\n>"
    )

    # Describe the image
    try:
        description = describe_image(args.image_path, model_id, prompt)
        print("Image Description:", description)
    except Exception as e:
        print(f"Error processing image: {e}")

# Run the script
if __name__ == "__main__":
    main()
