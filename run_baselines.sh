#!/bin/sh

export CUDA_VISIBLE_DEVICES=2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# RETRIEVER_NAME="bge"
# MODEL_PATH="/workspace/Search-R1/models/llama-3.1-8b-instruct"

# ################################-----musique-----######################################

# DATASET_NAME="musique"

# python ./pipelines/rag_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 21 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/searcho1_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_search_limit 7 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/react_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/tree_pipeline1.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# ##############################-----hotpotqa-----######################################
# DATASET_NAME="hotpotqa"

# python ./pipelines/rag_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 21 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/searcho1_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_search_limit 7 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/react_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/tree_pipeline1.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"


# ################################-----2wiki-----######################################

# DATASET_NAME="2wiki"

# python ./pipelines/rag_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 21 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/searcho1_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_search_limit 7 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/react_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/tree_pipeline1.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"


RETRIEVER_NAME="bge"
MODEL_PATH="/workspace/Search-R1/models/qwen3-8b"

# ################################-----musique-----######################################
# DATASET_NAME="musique"

# python ./pipelines/rag_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 21 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/searcho1_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_search_limit 7 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/react_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_turn 7 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

# python ./pipelines/tree_pipeline1.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

# python ./pipelines/rag_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 21 \
#     --output_dir "./outputs" \
#     --log_dir "./logs"

python ./pipelines/searcho1_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_search_limit 7 \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python ./pipelines/react_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"
    
python ./pipelines/tree_pipeline1.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --output_dir "./outputs" \
    --log_dir "./logs"


################################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 21 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python ./pipelines/searcho1_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_search_limit 7 \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python ./pipelines/react_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_turn 7 \
    --output_dir "./outputs" \
    --log_dir "./logs"

python ./pipelines/tree_pipeline1.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --output_dir "./outputs" \
    --log_dir "./logs"

