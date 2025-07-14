#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RETRIEVER_NAME="e5"
MODEL_PATH="/workspace/self-rag/model"
##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/selfrag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --output_dir "./outputs" \
    --log_dir "./logs"

################################------2wiki------######################################
DATASET_NAME="2wiki"

python ./pipelines/selfrag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --output_dir "./outputs" \
    --log_dir "./logs"

################################------musique------######################################
DATASET_NAME="musique"

python ./pipelines/selfrag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --output_dir "./outputs" \
    --log_dir "./logs"