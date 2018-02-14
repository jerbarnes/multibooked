MultiBooked: A corpus of Basque and Catalan Hotel Reviews Annotated for Aspect-level Sentiment Classification
==============

This is the finalized version of the corpora described in the following paper:

Jeremy Barnes, Patrik Lambert, and Toni Badia. 2016. **MultiBooked: A corpus of Basque and Catalan Hotel Reviews Annotated for Aspect-level Sentiment Classification**. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC'18)*.

The corpora are compiled from hotel reviews taken mainly from booking.com. The corpora are in Kaf/Naf format [https://github.com/opener-project/kaf/wiki/KAF-structure-overview] [https://github.com/newsreader/NAF], which is an xml-style stand-off format that allows for multiple layers of annotation. Each review was sentence- and word-tokenized and lemmatized using Freeling [http://nlp.lsi.upc.edu/freeling/node/1] for Catalan and ixa-pipes [http://ixa2.si.ehu.es/ixa-pipes/] for Basque. Finally, for each language two annotators annotated opinion holders, opinion targets, and opinion expressions for each review, following the guidelines set out in the OpeNER project [http://www.opener-project.eu/]. Details can be found in the paper.

This package includes the two corpora, as well as providing scripts to obtain corpus statistics (corpus_stats.py), reproduce the benchmarks reported in the paper (crf.py), extract only the opinionated units from the text (extract_opinions.py), or map the aspect-level annotations to sentence- or document-level annotated corpora (extract_sentences.py).


If you use these corpora for academic research, please cite the paper in question:
```
@inproceedings{Barnes2018multibooked,
    author={Barnes, Jeremy and Lambert, Patrik and Badia, Toni},
    title={MultiBooked: A corpus of Basque and Catalan Hotel Reviews Annotated for Aspect-level Sentiment Classification},
    booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC'18)},
    year = {2018},
    month = {May},
    date = {7-12},
    address = {Miyazaki, Japan},
    publisher = {European Language Resources Association (ELRA)},
    language = {english}
    }
```


Requirements for stats and extraction
--------
- Python 3
- NumPy

Additional Requirements for benchmarking
--------
- nltk [http://www.nltk.org/]
- sklearn [http://scikit-learn.org/stable/]
- sklearn-crfsuite [https://sklearn-crfsuite.readthedocs.io/en/latest/]	


Usage
--------

If you want to reproduce the results reported in the paper, simply clone the repository and run the experiment script:

```
git clone https://github.com/jbarnesspain/multibooked
cd multibooked
./run.sh
```

The script will reproduce the statistics and the benchmarks reported in the original paper.

```
================ Corpus Statistics ================
corpora/ca
Number of KAFs analyzed:  567
Average length of review: 45.0

Opinion Count: 2762
Targets:       2346
Holders:       236
--------------------------------------------------
corpora/eu
Number of KAFs analyzed:  343
Average length of review: 46.9

Opinion Count: 2328
Targets:       1775
Holders:       296
--------------------------------------------------

=================== Benchmarks ===================
Corpus: ca
Target F1:     0.66
Expression F1: 0.51
Holder F1:     0.58
F1 for classification: 0.77

Corpus: eu
Target F1:     0.54
Expression F1: 0.53
Holder F1:     0.47
F1 for classification: 0.85

```

If you want to extract the opinion-, sentence-, and document-level corpora, run the extraction script:
```
./extract.sh

```

The new corpora will be located in the 'corpora' directory.


License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
