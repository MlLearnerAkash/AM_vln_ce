import os
import sys
import numpy as np
import torch
import yaml
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from models.unet_decoder import QwenVLHeatmapDecoder
from models.qwen_vl import get_response_qwen_episode

class QwenVLHeatmapModel:
    """
    Wrapper model that takes an instruction and image sequence,
    gets Qwen2.5VL hidden states, and decodes them to a heatmap.
    """
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = self.config["device"]["device"]
        self.vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["model"]["model_path"],
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map=None
        ).to(self.device).eval()

        #NOTE: freezing the params- setting params to false
        for param in self.vl_model.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(self.config["model"]["model_path"])
        self.processor.tokenizer.padding_side = "left"

        self.decoder = QwenVLHeatmapDecoder(hidden_dim=3584, out_size=(640, 480)).to(self.device)

    def forward(self, instruction, image_sequence, heatmap_sequence, current_idx, history_length=3):
        with torch.no_grad():
            hidden_state = get_response_qwen_episode(
                model=self.vl_model,
                processor=self.processor,
                instruction=instruction,
                image_sequence=image_sequence,
                heatmap_sequence=heatmap_sequence,
                current_idx=current_idx,
                config=self.config,
                history_length=history_length,
                output_hidden_states=True
            )  # shape: (1, n_tokens, 3584)
            hidden_state= hidden_state.float()

        heatmap = self.decoder(hidden_state)  # shape: (1, 1, 640, 480)
        return heatmap
    
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    config_path = "/data/ws/VLN-CE/models/configs/train.yaml"
    model = QwenVLHeatmapModel(config_path)

    total_params, trainable_params = count_parameters(model.vl_model)
    print(f"Total parameters in decoder: {total_params}")
    print(f"Trainable parameters in decoder: {trainable_params}")

    instruction = "Go to the kitchen and find a cup on the table."
    image_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
    heatmap_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
    current_timestep = 2
    history_length = 1

    heatmap = model.forward(
        instruction=instruction,
        image_sequence=image_sequence,
        heatmap_sequence=heatmap_sequence,
        current_idx=current_timestep,
        history_length=history_length
    )
    print(f"Heatmap shape: {heatmap.shape}")  # (1, 1, 640, 480)

