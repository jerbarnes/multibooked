import sklearn_crfsuite
from sklearn_crfsuite import metrics
from utils import *
from corpus_stats import *
from nltk import PerceptronTagger
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold

label_dict = { 'B-StrongPositive': 'B-SP',
               'I-StrongPositive': 'I-SP',
               'B-Negative':'B-N',
               'I-Negative':'I-N',
               'B-Positive':'B-P',
               'I-Positive':'I-P',
               'B-StrongNegative':'B-SN',
               'I-StrongNegative':'I-SN'}

class_label_dict = { 'StrongPositive': 3,
                     'Positive': 2,
                     'Negative': 1,
                     'StrongNegative': 0
                   }

def convert_expressions_to_iob(kaf, tags=['exp', 'hol', 'tar']):
    """
    Converts a kaf/naf file to IOB labeled sentences.
    """
    exp = ['O'] * kaf.num_tokens
    for opinion in kaf.opinions:
        if 'exp' in tags:
            if opinion.opinion_expression:
                for i , (idx, p) in enumerate(opinion.opinion_expression):
                    if i == 0:
                        tag = label_dict['B-' + p]
                    else:
                        tag = label_dict['I-' + p]
                    try:
                        exp[idx -1] = tag
                    except IndexError:
                        pass
        if 'hol' in tags:
            if opinion.opinion_holder:
                for i, idx in enumerate(opinion.opinion_holder):
                    if i == 0:
                        tag = 'B-holder'
                    else:
                        tag = 'I-holder'
                    exp[idx-1] = tag
        if 'tar' in tags:
            if opinion.opinion_target:
                for i, idx in enumerate(opinion.opinion_target):
                    if i == 0:
                        tag = 'B-target'
                    else:
                        tag = 'I-target'
                    exp[idx-1] = tag
    return list(zip(list(kaf.tokens.values()), exp))

def convert_expressions_labels(kaf, label_dictionary):
    """
    Extracts only the opinion expressions and labels
    from the kaf file.
    """
    opinions = []
    for op in kaf.opinions:
        if op.opinion_expression:
            exp = [kaf.tokens[word].lower() for word, label in op.opinion_expression]
            label = label_dictionary[op.opinion_expression[0][1]]
            opinions.append((exp, label))
    return opinions

def word2features(sent, i):
    """
    Creates a feature vector for word i in sent.
    :param sent: a list of word, pos_tag tuples, ex: [("the", 'DET'), ("man",'NOUN'), ("ran",'VERB')]
    :param i   : an integer that refers to the index of the word in the sentence
    
    This function returns a dictionary object of features of the word at i.

    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    """
    Converts a sentence (list of (word, pos_tag) tuples) into a list of feature
    dictionaries
    """
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """
    Returns a list of labels, given a list of (token, pos_tag, label) tuples
    """
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    """
    Returns a list of tokens, given a list of (token, pos_tag, label) tuples
    """
    return [token for token, postag, label in sent]

def BOW(sent, w2idx):
    """
    Creates a bag of words representation of a sentence
    given a word, index dictionary
    """
    vec = np.zeros(len(w2idx))
    for w in sent:
        try:
            vec[w2idx[w]] = 1
        except KeyError:
            vec[0] = 1
    return vec

def get_vocab(expressions):
    vocab = {}
    vocab['UNK'] = 0
    for exp, label in expressions:
        for w in exp:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

if __name__ == '__main__':

    for corpus in os.listdir('corpora'):
        sents = []
        only_expressions = []
        if corpus == 'eu':
            ftype = 'naf'
            tagger = PerceptronTagger(load=False)
            tagger.load('models/averaged_perceptron_basque_tagger.pkl')
        else:
            ftype = 'kaf'
            tagger = PerceptronTagger(load=False)
            tagger.load('models/averaged_perceptron_catalan_tagger.pkl')
            
        for file in os.listdir(os.path.join('corpora', corpus)):
            try:
                kaf = KafNafOpinion(open(os.path.join('corpora', corpus, file)).read(), ftype)
                bio = convert_expressions_to_iob(kaf, ['exp', 'hol', 'tar'])
                tagged = tagger.tag([w for w, t in bio])
                sents.append([(w, p, t) for (w, p), (w, t) in zip(tagged, bio)])

                opinions = convert_expressions_labels(kaf, class_label_dict)
                only_expressions.extend(opinions)
                
            except IndexError:
                pass

        sents = np.array(sents)
        only_expressions = np.array(only_expressions)

        # Create cross validation splits
        target_f1s = []
        expression_f1s = []
        holder_f1s = []

        print('Corpus: {0}'.format(corpus))
        print('Performing 10-fold cross-validation...')

        labels = ['B-target', 'I-target', 'B-holder', 'B-SP', 'I-SP', 'B-P', 'I-P', 'B-N', 'I-N', 'B-SN', 'I-SN']
        target_labels = ['B-target', 'I-target']
        expression_labels = ['B-SP', 'I-SP', 'B-P', 'I-P', 'B-N', 'I-N', 'B-SN', 'I-SN']
        holder_labels = ['B-holder']

        kf = KFold(len(sents), n_folds=10)
        for train_index, test_index in kf:
            train_sents, test_sents = sents[train_index], sents[test_index]

            # Extract features for training CRF
            X_train = [sent2features(s) for s in train_sents]
            y_train = [sent2labels(s) for s in train_sents]
            X_test = [sent2features(s) for s in test_sents]
            y_test = [sent2labels(s) for s in test_sents]

            # Setup CRF classifier
        
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True)

            # Train classifier
            crf.fit(X_train, y_train)
            y_pred = crf.predict(X_test)


            target_f1s.append(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=target_labels))
            expression_f1s.append(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=expression_labels))
            holder_f1s.append(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=holder_labels))
        

        # Create cross-validations splits for classification
        classification_f1s = []

        kf = KFold(len(only_expressions), n_folds=10)
        for train_index, test_index in kf:
            train_exp, test_exp = only_expressions[train_index], only_expressions[test_index]

            # Get word2idx dictionary
            w2idx = get_vocab(only_expressions)

            # Create BOW reps
            exp_X_train = [BOW(exp, w2idx) for exp, label in train_exp]
            exp_y_train = [label for exp, label in train_exp]

            exp_X_test = [BOW(exp, w2idx) for exp, label in test_exp]
            exp_y_test = [label for exp, label in test_exp]

            # Train and test Linear SVM on BOW reps for classification
            clf = LinearSVC()
            clf.fit(exp_X_train, exp_y_train)
            pred = clf.predict(exp_X_test)
            classification_f1s.append(f1_score(exp_y_test, pred,
                                               labels=sorted(set(exp_y_test)),
                                               average='weighted'))

        print('Target F1:     {0:.2f}'.format(sum(target_f1s) / len(target_f1s)))
        print('Expression F1: {0:.2f}'.format(sum(expression_f1s) / len(expression_f1s)))
        print('Holder F1:     {0:.2f}'.format(sum(holder_f1s) / len(holder_f1s)))

        print('F1 for classification: {0:.2f}'.format(sum(classification_f1s) / len(classification_f1s)))
        print()
