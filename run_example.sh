#!/bin/bash

dataset_name=UMLs
date_stamp=$(date +"%Y%m%d")
api_key=API_KEY_HERE
llm_name=GPT35
run_name=${date_stamp}_${dataset_name}_${llm_name}

echo "an simple example"

date; echo "run extractor"; echo
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name} --run_extractor --rand_seed ${rseed}

date; echo "run proposer"; echo
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name}  --run_proposer --llm_name ${llm_name} --llm_max_input_chars 4096 --llm_api_key ${api_key} 

date; echo "run reasoner"; echo
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name}  --run_reasoner --kge_bsize 32