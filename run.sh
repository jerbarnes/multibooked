#!/bin/bash

# get corpus statistics
echo "================ Corpus Statistics ================"
python3 corpus_stats.py -dir corpora/ca -type kaf
python3 corpus_stats.py -dir corpora/eu -type naf
echo

# reproduce benchmarks
echo "=================== Benchmarks ==================="
python3 benchmark.py