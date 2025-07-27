#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CONFIG_PATH="/workspace/Search-R1/config/dataset_paths.json"

RETRIEVER_NAME="bge"
TOPK=5



###################################-----llama3.1-70b--awq-----######################################

MODEL_PATH="/workspace/Search-R1/models/llama-3.1-70b-instruct"

################################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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




################################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


################################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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




##################################-----llama-3.1-8b-instruct-----######################################

MODEL_PATH="/workspace/Search-R1/models/llama-3.1-8b-instruct"

################################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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



################################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


################################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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



###################################-----qwen3-8b-----######################################


MODEL_PATH="/workspace/Search-R1/models/qwen3-8b"

################################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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



################################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


################################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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





###################################-----qwen3-32b-awq-----######################################

MODEL_PATH="/workspace/Search-R1/models/qwen3-32b-awq"

################################-----musique-----######################################
DATASET_NAME="musique"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


##############################-----hotpotqa-----######################################
DATASET_NAME="hotpotqa"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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



################################-----2wiki-----######################################
DATASET_NAME="2wiki"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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


################################-----bamboogle-----######################################
DATASET_NAME="bamboogle"

python ./pipelines/naive_pipeline.py \
    --model_path $MODEL_PATH \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --output_dir "./outputs" 

python ./pipelines/rag_pipeline.py \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
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

python ./pipelines/react_pipeline.py \
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

