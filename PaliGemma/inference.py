import torch
import fire
from PIL import Image


from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model


def test_inference(
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        device: str,
        prompt: str,
        image_file_path: str,
        max_tokens_to_generate: int,
        temperature: float,
        top_p: float,
        do_sample: bool,

):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    # Process the input prompt and image
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    
   




def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,

):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print(f"Using device: {device}")

    print ("Loading model...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )




if __name__ == "__main__":
    fire.Fire(main)

