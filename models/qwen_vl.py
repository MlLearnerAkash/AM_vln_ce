from qwen_vl_utils import process_vision_info
from PIL import Image
import yaml
import numpy as np

import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def preprocess_image_nearest(image_input: str, image_resolution: int, perform_resize: bool = False) -> tuple[int, int]:
    try:
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            raise ValueError("Input must be a numpy array or a file path.")

        original_width, original_height = image.width, image.height
        max_dimension = max(original_width, original_height)

        if max_dimension > image_resolution:
            resize_factor = image_resolution / max_dimension
            new_width = int(original_width * resize_factor)
            new_height = int(original_height * resize_factor)
        else:
            new_width, new_height = original_width, original_height

        if perform_resize:
            resized_image = image.resize((new_width, new_height), resample=Image.NEAREST)
            if image.mode != "RGB":
                resized_image = resized_image.convert("RGB")
            return new_width, new_height, resized_image

    except Exception as e:
        raise IOError(f"Can not open the image {image_input}: {e}")

    return new_height, new_width



def build_message(prompt, config, current_image_path, past_heatmap_paths):
    """
    Build the message to send to the model with current image and past heatmaps.
    
    Args:
        prompt (str): Text prompt (instruction for the episode)
        config: Configuration object
        current_image_path (str): Path to current RGB image at timestep t
        past_heatmap_paths (list): List of paths to past heatmap images from t-k to t-1
    """
    user_prompt = []
    
    # Add current image (at timestep t)
    train_height, train_width = preprocess_image_nearest(current_image_path, config["dataset"]["img_res"])
    user_prompt.append({
        "type": "image",
        "image": current_image_path,
        "resized_height": train_height,
        "resized_width": train_width
    })
    
    # Add past heatmaps (history from t-k to t-1)
    for heatmap_path in past_heatmap_paths:
        train_height, train_width = preprocess_image_nearest(heatmap_path, config["dataset"]["img_res"])
        user_prompt.append({
            "type": "image",
            "image": heatmap_path,
            "resized_height": train_height,
            "resized_width": train_width
        })
    
    # Add text prompt (episode instruction)
    user_prompt.append({"type": "text", "text": prompt})
    
    message = [{"role": "user", "content": user_prompt}]
    return message

def get_response_qwen_episode(model, processor, instruction, image_sequence, heatmap_sequence, current_idx, config, history_length=3, output_hidden_states=True):
    """
    Process a single episode at timestep t with instruction and image history.
    
    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor
        instruction (str): Single instruction for the entire episode
        image_sequence (list): List of all RGB image paths in the episode [img_0, img_1, ..., img_T]
        heatmap_sequence (list): List of all heatmap paths in the episode [hmap_0, hmap_1, ..., hmap_T]
        current_idx (int): Current timestep t (0 <= t < len(image_sequence))
        config: Configuration object
        history_length (int): Number of past frames to include (default: 3)
        output_hidden_states (bool): If True, return last hidden state instead of generated text
    
    Returns:
        torch.Tensor or list: Last hidden state or generated text
    """
    # Get current image at timestep t
    current_image_path = image_sequence[current_idx]
    
    # Get past heatmaps from t-history_length to t-1
    past_heatmap_paths = []
    for i in range(max(0, current_idx - history_length), current_idx):
        past_heatmap_paths.append(heatmap_sequence[i])
    
    # Build message with instruction and image/heatmap sequence
    message = build_message(instruction, config, current_image_path, past_heatmap_paths)
    
    # Preparation for inference
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(config["device"]["device"])
    
    if output_hidden_states:
        # Forward pass to get last hidden state
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state
    else:
        # Generate text response
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        output_text = processor.decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


if __name__ == "__main__":
    

    with open("/data/ws/VLN-CE/models/configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading model from {config['model']['model_path']}")
    vmodel = Qwen2VLForConditionalGeneration.from_pretrained(
        config["model"]["model_path"], torch_dtype="auto", attn_implementation="flash_attention_2", device_map=None
    )
    vmodel.eval()
    vmodel.to(config["device"]["device"])

    processor = AutoProcessor.from_pretrained(config['model']['model_path'])
    processor.tokenizer.padding_side = "left"

    episode_instruction = "Go to the kitchen and find a cup on the table."


    image_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
    heatmap_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
    
    
    current_timestep = 2
    history_length = 1
    
    hidden_state = get_response_qwen_episode(
        model=vmodel,
        processor=processor,
        instruction=episode_instruction,
        image_sequence=image_sequence,
        heatmap_sequence=heatmap_sequence,
        current_idx=current_timestep,
        config=config,
        history_length=history_length,
        output_hidden_states=True
    )
    print(hidden_state)#(1, 157, 3584)

