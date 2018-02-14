#!/bin/bash

# extract opinion-level corpora
python3 extract_opinions.py -indir corpora/ca/ -outdir corpora/ca_opinions -type kaf
python3 extract_opinions.py -indir corpora/eu/ -outdir corpora/eu_opinions -type naf

# extract sentence-level corpora
python3 extract_sentences.py -indir corpora/ca/ -outdir corpora/ca_sents -type kaf -level sent
python3 extract_sentences.py -indir corpora/eu/ -outdir corpora/eu_sents -type naf -level sent

# extract document-level corpora
python3 extract_sentences.py -indir corpora/ca/ -outdir corpora/ca_docs -type kaf -level docs
python3 extract_sentences.py -indir corpora/eu/ -outdir corpora/eu_docs -type naf -level docs