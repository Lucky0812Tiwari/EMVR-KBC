# LLM-enhanced Symbolic Reasoning for KBC

## Introduction

This repository provides resources for paper “Large Language Model-Enhanced Symbolic Reasoning for Knowledge Base Completion”.

## Dependencies

- Python 3.9.19
- transformers==4.40.1
- scikit-learn==1.4.2
- scipy==1.13.0
- torch==2.3.0
- transformers==4.40.1
- nltk==3.8.1
- sentence-transformers==3.0.0
- sentencepiece==0.2.0
- openai==1.24.0
- google-generativeai==0.7.2
- groq==0.31.0

## Code Files

The dataset for UMLs/WN18RR/FB15K found [here](https://github.com/DeepGraphLearning/RNNLogic), CN100 could be found [here](https://home.ttic.edu/~kgimpel/commonsense.html) and WD15K could be found [here](https://github.com/THU-KEG/BIMR) with the interpretability annotations. RotatE could be found [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/) and `kge.py` includes part of the said implementation. The main python file is `lesr.py`. We include the FB15K relation mapping in `fb15k_rels.csv` and example commands in `run_example.sh`. To run LeSR, please create subdirectories `data/`, `log/`, `runs/`, move dataset and kge (if used) to their respective folder, and provide own LLM inference api key. To use other knowledge base data, please edit the commandline parse and add data reading in `data.py`.