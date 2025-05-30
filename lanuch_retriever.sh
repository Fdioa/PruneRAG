index_file=/share/datasets/data_wiki_index_hnsw64/e5_HNSW64.index
corpus_file=/workspace/Search-R1/corpus1/wiki-18.jsonl
retriever_name=e5
retriever_path=/workspace/Search-R1/model

python scripts/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 1 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path 
