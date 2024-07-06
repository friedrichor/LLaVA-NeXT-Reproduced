#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/v1_6/eval/mmbench.sh
### !/bin/bash
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

set -x
export PYTHONPATH=/mmu_nlp_hdd/kongfanheng/train_code/FastChat
which python

gpu_list="1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV_MODE=v1
CKPT="llava-v1.6-7b_vicuna-sft_mix1M"
CKPT_DIR="checkpoints"
SPLIT="mmbench_dev_20230712"
LANG="en"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang en \
        --single-pred-prompt \
        --square_eval True \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT
mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT} \
    --experiment merge
