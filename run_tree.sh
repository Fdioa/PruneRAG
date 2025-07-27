#!/bin/sh
# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn


CONFIG_PATH="/workspace/Search-R1/config/dataset_paths.json"

RETRIEVER_NAME="bge"
TOPK=5


MODEL_PATH="/workspace/Search-R1/models/llama-3.1-8b-instruct"
##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"





##################################-----qwen3-8b-----######################################


MODEL_PATH="/workspace/Search-R1/models/qwen3-8b"
##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"






##################################-----qwen3-32b-awq-----######################################


MODEL_PATH="/workspace/Search-R1/models/qwen3-32b-awq"
##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"




