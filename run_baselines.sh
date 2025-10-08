#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CONFIG_PATH="./config/dataset_paths.json"

RETRIEVER_NAME="e5"
TOPK=5

##################################-----llama-3.1-8b-instruct-----######################################

MODEL_PATH="./models/llama-3.1-8b-instruct"

################################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python -m pipelines.naive_pipeline \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python -m pipelines.rag_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --output_dir "./outputs" \
    --log_dir "./logs"

python -m pipelines.memorag_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --output_dir "./outputs" \
    --log_dir "./logs"

python -m pipelines.react_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python ./pipelines/searcho1_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_search_limit 7 \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"


python -m pipelines.conregen_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 2 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python -m pipelines.probtree_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --output_dir "./outputs" \
    --log_dir "./logs"
