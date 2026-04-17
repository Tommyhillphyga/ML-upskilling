MODEL_PATH = "paligemma-3b-pt-224"     # Clone/download the model repo on huggingface and add the path here. 
PROMPT = "What is in this image?"
IMAGE_FILE_PATH = "test_image/test_image.jpg"
MAX_TOKEN_TO_GENERATE = 100
TEMPERATURE = 0.8
TOP_P = 0.90
DO_SAMPLE = "False"
ONLY_CPU = "False"


python inference.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --image_file_path $IMAGE_PATH \
    --max_token_to_generate $MAX_TOKEN_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU