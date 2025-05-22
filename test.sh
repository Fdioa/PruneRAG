bin/bash

python ./search_r1/llm_agent/native_pipeline.py \
    --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --output_dir "./output/2wki" \

python ./search_r1/llm_agent/rag_pipeline.py \
    --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --topk 3 \
    --max_context_length 4096 \
    --output_dir "./output/2wiki" \
    --log_dir "./logs/2wiki"

python ./search_r1/llm_agent/tree_pipeline.py \
    --model_path "/workspace/Search-R1/models/llama-3.1-8b-instruct" \
    --retrieval_url "http://localhost:8000" \
    --data_path "/workspace/Search-R1/config/dataset_paths.json" \
    --dataset_name "2wiki" \
    --split "test" \
    --topk 3 \
    --max_depth 3 \
    --max_context_length 4096 \
    --output_dir "./output/2wiki" \
    --log_dir "./logs/2wiki"



    





