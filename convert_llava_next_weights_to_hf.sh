### !/bin/bash

python convert_llava_next_weights_to_hf.py \
    --local_model_path ./checkpoints/llava-next-vicuna-7b-sft \
    --pytorch_dump_folder_path ./checkpoints/llava-next-vicuna-7b-sft-hf \
    --push_to_hub
