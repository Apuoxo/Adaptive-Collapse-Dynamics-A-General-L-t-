#!/bin/bash

# Train utility MLP
python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf --epochs 3

# Run PG19 evaluation
python experiments/run_pg19.py --cache-budget 1024 --output results/pg19_ace.json

# Run Needle-in-a-Haystack
python experiments/run_needle_test.py --haystack-len 32000 --needle-depth 31000

# Run LongBench
python experiments/run_longbench.py --tasks narrativeqa,multifieldqa,trec

echo "All evaluations complete!"