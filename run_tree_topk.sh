#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RETRIEVER_NAME="e5"
MODEL_PATH="/workspace/Search-R1/models/qwen3-8b"
CONFIG_PATH="/workspace/Search-R1/config/dataset_paths.json"

# ###############################--topk=1--#######################################################

# ##############################-----hotpotqa-----######################################
# DATASET_NAME="hotpotqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----musique-----######################################
# DATASET_NAME="musique"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----bamboogle-----######################################
# DATASET_NAME="bamboogle"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----2wiki-----######################################
# DATASET_NAME="2wiki"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----nq-----######################################
# DATASET_NAME="nq"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----triviaqa-----######################################
# DATASET_NAME="triviaqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 1 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"





# ###############################--topk=3--#######################################################

# ##############################-----hotpotqa-----######################################
# DATASET_NAME="hotpotqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----musique-----######################################
# DATASET_NAME="musique"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----bamboogle-----######################################
# DATASET_NAME="bamboogle"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----2wiki-----######################################
# DATASET_NAME="2wiki"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----nq-----######################################
# DATASET_NAME="nq"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----triviaqa-----######################################
# DATASET_NAME="triviaqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"






# ###############################--topk=5--#######################################################

# ##############################-----hotpotqa-----######################################
# DATASET_NAME="hotpotqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----musique-----######################################
# DATASET_NAME="musique"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----bamboogle-----######################################
# DATASET_NAME="bamboogle"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----2wiki-----######################################
# DATASET_NAME="2wiki"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----nq-----######################################
# DATASET_NAME="nq"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----triviaqa-----######################################
# DATASET_NAME="triviaqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 5 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"











# ###################################--topk=7--#######################################################


# ##############################-----hotpotqa-----######################################
# DATASET_NAME="hotpotqa"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 7 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----musique-----######################################
# DATASET_NAME="musique"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 7 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----bamboogle-----######################################
# DATASET_NAME="bamboogle"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 7 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"

# ##############################-----2wiki-----######################################
# DATASET_NAME="2wiki"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 7 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


# ##############################-----nq-----######################################
# DATASET_NAME="nq"

# python ./pipelines/tree_pipeline.py \
#     --model_path $MODEL_PATH \
#     --retriever_name $RETRIEVER_NAME \
#     --retrieval_url "http://localhost:8000" \
#     --data_path $CONFIG_PATH \
#     --dataset_name $DATASET_NAME \
#     --split "test" \
#     --topk 7 \
#     --max_depth 3 \
#     --all_decom_depth 0 \
#     --threshold 0.9 \
#     --output_dir "./outputs_topk" \
#     --log_dir "./logs_topk"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 7 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"












###################################--topk=10--#######################################################


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"

##############################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"

##############################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"

##############################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"


##############################-----nq-----######################################
DATASET_NAME="nq"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"


##############################-----triviaqa-----######################################
DATASET_NAME="triviaqa"

python ./pipelines/tree_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 10 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.9 \
    --output_dir "./outputs_topk" \
    --log_dir "./logs_topk"



