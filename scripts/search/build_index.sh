
corpus_file=/workspace/Search-R1/corpus/wiki-18.jsonl # jsonl
save_dir=/workspace/index/qwen3_index
retriever_name=qwen3 # this is for indexing naming
retriever_model=/workspace/Search-R1/models/qwen3-embedding-0.6b

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1 python ./scripts/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
