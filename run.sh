#!/bin/sh

##############################-----hotpotqa-----######################################

python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "hotpotqa" \
    --split "test" \
    --output_dir "./output/hotpotqa" 

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "hotpotqa" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/hotpotqa" \
    --log_dir "./logs/hotpotqa"

# python ./search_r1/llm_agent/tree_pipeline.py \
#     --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "hotpotqa" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --output_dir "./output/hotpotqa" \
#     --log_dir "./logs/hotpotqa"

# python ./search_r1/llm_agent/searcho1_pipeline.py \
#     --model_path "/workspace/Search-o1/models/qwq_awq" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "hotpotqa" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --max_search_limit 7 \
#     --output_dir "./output/hotpotqa" \
#     --log_dir "./logs/hotpotqa"


################################-----2wiki-----######################################

python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --output_dir "./output/2wiki" 

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/2wiki" \
    --log_dir "./logs/2wiki"

# python ./search_r1/llm_agent/tree_pipeline.py \
#     --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "2wiki" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --output_dir "./output/2wiki" \
#     --log_dir "./logs/2wiki"

python ./search_r1/llm_agent/searcho1_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --max_context_length 4096 \
    --max_search_limit 7 \
    --output_dir "./output/2wiki" \
    --log_dir "./logs/2wiki"


###############################-----nq-----######################################

python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "nq" \
    --split "test" \
    --output_dir "./output/nq" 

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "nq" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/nq" \
    --log_dir "./logs/nq"

# python ./search_r1/llm_agent/tree_pipeline.py \
#     --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "nq" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --output_dir "./output/nq" \
#     --log_dir "./logs/nq"



python ./search_r1/llm_agent/searcho1_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "nq" \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --max_context_length 4096 \
    --max_search_limit 7 \
    --output_dir "./output/nq" \
    --log_dir "./logs/nq"


##############################-----triviaqa-----######################################


python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "triviaqa" \
    --split "test" \
    --output_dir "./output/triviaqa" 

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "triviaqa" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/triviaqa" \
    --log_dir "./logs/triviaqa"

# python ./search_r1/llm_agent/tree_pipeline.py \
#     --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "triviaqa" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --output_dir "./output/triviaqa" \
#     --log_dir "./logs/triviaqa"


python ./search_r1/llm_agent/searcho1_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "triviaqa" \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --max_context_length 4096 \
    --max_search_limit 7 \
    --output_dir "./output/triviaqa" \
    --log_dir "./logs/triviaqa"


##############################-----musique-----######################################

python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "musique" \
    --split "test" \
    --output_dir "./output/musique" 

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "musique" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/musique" \
    --log_dir "./logs/musique"

# python ./search_r1/llm_agent/tree_pipeline.py \
#     --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
#     --retrieval_url "http://localhost:8000" \
#     --data_path "/workspace/Search-R1/config/dataset_paths.json" \
#     --dataset_name "musique" \
#     --split "test" \
#     --topk 3 \
#     --max_depth 3 \
#     --max_context_length 4096 \
#     --output_dir "./output/musique" \
#     --log_dir "./logs/musique"


python ./search_r1/llm_agent/searcho1_pipeline.py \
    --model_path "/workspace/Search-o1/models/qwq_awq" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "musique" \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --max_context_length 4096 \
    --max_search_limit 7 \
    --output_dir "./output/musique" \
    --log_dir "./logs/musique"

