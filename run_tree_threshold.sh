#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RETRIEVER_NAME="bge"
MODEL_PATH="/workspace/Search-R1/models/qwen3-8b"
CONFIG_PATH="/workspace/Search-R1/config/dataset_paths.json"

##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0\
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5\
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.5 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.55 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.6 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.65 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.7 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"



##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.75 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"




##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"
##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.8 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.85 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"



##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.92 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"



##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.94 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"



##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.96 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.98 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 1 \
    --output_dir "./outputs_threshold" \
    --log_dir "./logs_threshold"

