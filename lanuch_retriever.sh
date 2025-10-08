export CUDA_VISIBLE_DEVICES=0,1
corpus_file=/workspace/PruneRAG/corpus/wiki-18.jsonl

index_file=/share/datasets/data_wiki_index_flat/e5_Flat.index
retriever_name=e5
retriever_path=/workspace/PruneRAG/models/e5-base-v2

# index_file=/workspace/index/bge_index/bge_Flat.index
# retriever_name=bge
# retriever_path=/workspace/PruneRAG/models/bge-large-en-v1.5

python ./scripts/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
