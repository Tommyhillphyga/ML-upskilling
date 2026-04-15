from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
from safetensors import safe_open
from typing import Tuple

import os
import json
import glob

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    #load the model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files in the model directory
    safetensors_files = glob.glob(os.path.join(model_path, padding_side = "right"))

    # ... and load them one by one in the tensor dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device = "cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

   # load the model config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # create the model using the configuration 
    model = PaliGemmaForConditionalGeneration(config).to(device) #This is the modelel

    # load the model weights using the tensor dictionary
    model.load_state_dict(tensors, strict=False) # load the model weights using the tensor dictionary

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
