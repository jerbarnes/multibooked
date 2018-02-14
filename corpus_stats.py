import sys, os, re, copy
import numpy as np
import argparse

"""
Compares the annotations of different KAF files using
the agr function from:

    Janyce  Wiebe  and  Claire  Cardie.   2005.   Annotating
        expressions of opinions and emotions in language.
        language  resources  and  evaluation. In
        Language Resources and Evaluation (formerly
        Computers and the Humanities, page 2005.

For comparing two annotations in the BIO format,
which do not necessarily overlap. Kappa cannot be
applied, since we do not want to penalize annotations
that coincide partially.

Given two annotations A, B from annotators a and b:

agr(a||b) = |Matching from A and B| / |A|

This can be thought of as recall, when a is the gold standard
or precision if b is the gold standard. In order to get
an agreement score, we can take the average of agr
both ways.
"""

def get_between(left, right, text):
    no_left = re.split(left, text)[1]
    return re.split(right, no_left)[0].strip()

def get_token_section(KAF):
    return get_between('<text>', '</text>', KAF)

def get_tokens(token_section, type='kaf'):
    tokens = {}
    for tok in re.split('<wf', token_section):
        try:
            if type =='kaf':
                wid = re.findall('wid="w_([0-9]*)"', tok)[0]
                word = re.findall('sent="[0-9]*">(.*)</wf>', tok)[0]
            else:
                wid = re.findall('id="w([0-9]*)"', tok)[0]
                word = re.findall('para="[0-9]*">(.*)</wf>', tok)[0]
            tokens[int(wid)] = word
        except IndexError:
            pass
    return tokens

def get_num_tokens(KAF, type='kaf'):
    if type == 'kaf':
        return max([int(w) for w in re.findall('wid="w_([0-9]*)"', KAF)])
    else:
        return max([int(w) for w in re.findall('id="w([0-9]*)"', KAF)])
    
def get_opinion_section(annotation):
    """Gets opinion annotations from KAF file"""
    if len(re.findall('<opinions>', annotation)) > 0:    
        return get_between('<opinions>', '</opinions>', annotation)

def get_opinions(opinion_section):
    """Splits opinion section into seperate opinions"""
    if opinion_section != None:
        opinions = []
        for i in re.split('</opinion>', opinion_section):
            opinions.append(i)
        return opinions[:-1]
    else:
        return None

def get_opinion_id(opinion):
    """Gets tag id of opinion"""
    if opinion != None:
        tag_id = re.findall('<opinion oid="o([0-9]*)">', opinion)[0]
        return int(tag_id)
    else:
        return None

def get_opinion_holder(opinion, type='kaf'):
    """Gets word ids of opinion holder"""
    
    if len(re.findall('<opinion_holder>', opinion)) > 0:
        oh = get_between('<opinion_holder>', '</opinion_holder>', opinion)
        if type == 'kaf':
            oh_ids = re.findall('<target id="t_([0-9]*)"', oh)
        else:
            oh_ids = re.findall('<target id="t([0-9]*)"', oh)
        return [int(oh_id) for oh_id in oh_ids]
    else:
        return None

def get_opinion_target(opinion, type='kaf'):
    """Gets word ids of opinion target"""
    
    if len(re.findall('<opinion_target>', opinion)) > 0:
        ot = get_between('<opinion_target>', '</opinion_target>', opinion)
        if type == 'kaf':
            ot_ids = re.findall('<target id="t_([0-9]*)"', ot)
        else:
            ot_ids = re.findall('<target id="t([0-9]*)"', ot)
        return [int(ot_id) for ot_id in ot_ids]
    else:
        return None

def get_opinion_expression(opinion, type='kaf'):
    """Gets word ids and polarity of opinion (strong positive,
       positive, negative, or strong negative)."""
    
    if len(re.findall('<opinion_expression', opinion)) > 0:
        polarity = re.findall('<opinion_expression polarity="(.*)">', opinion)[0]
        oe = get_between('<opinion_expression', '</opinion_expression>', opinion)
        if type == 'kaf':
            oe_ids = [int(oe_id) for oe_id in re.findall('<target id="t_([0-9]*)"', oe)]
        else:
            oe_ids = [int(oe_id) for oe_id in re.findall('<target id="t([0-9]*)"', oe)]
        return [(oe_id, polarity) for oe_id in oe_ids]
    else:
        return None



class KafNafOpinion(object):

    def __init__(self, KAF, type='kaf'):
        self.KAF = KAF
        self.type = type
        self.tokens_section = get_token_section(self.KAF)
        self.tokens = get_tokens(self.tokens_section, type=self.type)
        self.tokens_2_idx = dict([(i,w) for w,i in self.tokens.items()])
        self.num_tokens = get_num_tokens(self.KAF,type=self.type)
        self.opinion_section = get_opinion_section(self.KAF)
        self.raw_opinions = get_opinions(self.opinion_section)
        self.opinions = self.parse_opinions(self.raw_opinions, self.num_tokens)


    def parse_opinions(self, raw_opinions, num_toks_in_doc):
        opinions = []
        if raw_opinions != None:
            for opinion in raw_opinions:
                opinion_id = get_opinion_id(opinion)
                opinion_holder = get_opinion_holder(opinion,type=self.type)
                opinion_target = get_opinion_target(opinion,type=self.type)
                opinion_expression = get_opinion_expression(opinion,type=self.type)
                this_op = Opinion(num_toks_in_doc, opinion_id, opinion_holder,
                                  opinion_target, opinion_expression)
                opinions.append(this_op)
        else:
            opinion_id = None
            opinion_holder = None
            opinion_target = None
            opinion_expression = None
            this_op = Opinion(num_toks_in_doc, opinion_id, opinion_holder,
                              opinion_target, opinion_expression)
            opinions.append(this_op)
        return opinions

    def print_opinions(self):
        for opinion in self.opinions:
            opinion.print_opinion(self.tokens)
            print('-'*40)

    def print_text(self):
        print(' '.join([self.tokens[i+1] for i in range(len(self.tokens))]))
        

class Opinion(object):

    def __init__(self, num_tokens_in_doc, opinion_id, opinion_holder,
                 opinion_target, opinion_expression):

        self.num_toks = num_tokens_in_doc
        self.opinion_id = opinion_id
        self.opinion_holder = opinion_holder
        self.opinion_target = opinion_target
        self.opinion_expression = opinion_expression

    def print_opinion(self, idx_to_token):
        print('Opinion Id: {0}'.format(self.opinion_id))
        if self.opinion_holder:
            print('Opinion_holder: {0}'.format(' '.join(
            [idx_to_token[i] for i in self.opinion_holder])))
        else:
            print('Opinion_holder: None')
        if self.opinion_target:
            print('Opinion_target: {0}'.format(' '.join(
            [idx_to_token[i] for i in self.opinion_target])))
        else:
            print('Opinion_target: None')
        print('Opinion_expression: {0}'.format(' '.join(
            [idx_to_token[i] for i, strength in self.opinion_expression])))
        print('Opinion_strength: {0}'.format(self.opinion_expression[0][1]))

def open_kafs(DIR):
    file_names = os.listdir(DIR)
    kafs = [open(os.path.join(DIR,fn)).read() for fn in sorted(file_names) if fn.endswith('.kaf')]
    return kafs

def compare_opinion_lists(List1, List2, error_function, type='kaf'):

    errors = []
    for kaf1,kaf2 in list(zip(List1, List2)):
        try:
            err = compare_opinion_expressions(kaf1, kaf2, error_function, type)
            if err:
                errors.append(err)
        except IndexError:
            pass

    return sum(errors)/float(len(errors))

def get_number_tokens(KAFS, type='kaf'):
    num_tokens = 0
    for kaf in KAFS:
        ko = KafNafOpinion(kaf, type)
        num_tokens += ko.num_tokens
    return num_tokens

def get_number_of_instances(KAFS, type='kaf'):

    opinion_count = 0
    opinion_target = 0
    opinion_holder = 0

    
    for kaf in KAFS:
        ko = KafNafOpinion(kaf, type)
        for opinion in ko.opinions:
            if opinion.opinion_expression != None:
                opinion_count += 1
            if opinion.opinion_target != None:
                opinion_target += 1
            if opinion.opinion_holder != None:
                opinion_holder += 1

    return opinion_count, opinion_target, opinion_holder


def agr_holder_lists(List1, List2, type='kaf'):

    errors = []
    for kaf1,kaf2 in list(zip(List1, List2)):
        try:
            err = agr_holders(kaf1, kaf2, type)
            errors.append(err)
        except IndexError:
            pass

    return sum(errors)/float(len(errors))

def agr_target_lists(List1, List2, type='kaf'):

    errors = []
    for kaf1,kaf2 in list(zip(List1, List2)):
        try:
            err = agr_targets(kaf1, kaf2, type)
            errors.append(err)
        except IndexError:
            pass

    return sum(errors)/float(len(errors))

def agr_expressions_lists(List1, List2, type='kaf'):

    errors = []
    for kaf1,kaf2 in list(zip(List1, List2)):
        try:
            err = agr_expressions(kaf1, kaf2, type)
            errors.append(err)
        except IndexError:
            pass

    return sum(errors)/float(len(errors))

def agr_targets(KAF1, KAF2, type='kaf'):
    ko1 = KafNafOpinion(KAF1, type)
    ko2 = KafNafOpinion(KAF2, type)
    
    opinion_targets1 = []
    opinion_targets2 = []

    for opinion in ko1.opinions:
        try:
            opinion_targets1.extend(opinion.opinion_target)
        except TypeError:
            pass
        
    for opinion in ko2.opinions:
        try:
            opinion_targets2.extend(opinion.opinion_target)
        except TypeError:
            pass
    o1 = set(opinion_targets1)
    o2 = set(opinion_targets2)
    agr_a_b = len(o1.intersection(o2)) / max(len(o1), 0.000001)
    agr_b_a = len(o2.intersection(o1)) / max(len(o2), 0.000001)
    return (agr_a_b + agr_b_a) / 2

def agr_expressions(KAF1, KAF2, type='kaf'):
    ko1 = KafNafOpinion(KAF1, type)
    ko2 = KafNafOpinion(KAF2, type)
    
    opinion_expressions1 = []
    opinion_expressions2 = []

    for opinion in ko1.opinions:
        try:
            opinion_expressions1.extend(opinion.opinion_expression)
        except TypeError:
            pass
        
    for opinion in ko2.opinions:
        try:
            opinion_expressions2.extend(opinion.opinion_expression)
        except TypeError:
            pass
    o1 = set(opinion_expressions1)
    o2 = set(opinion_expressions2)
    agr_a_b = len(o1.intersection(o2)) / max(len(o1), 0.000001)
    agr_b_a = len(o2.intersection(o1)) / max(len(o2), 0.000001)
    return (agr_a_b + agr_b_a) / 2

def agr_holders(KAF1, KAF2, type='kaf'):
    ko1 = KafNafOpinion(KAF1, type)
    ko2 = KafNafOpinion(KAF2, type)
    
    opinion_holders1 = []
    opinion_holders2 = []

    for opinion in ko1.opinions:
        try:
            opinion_holders1.extend(opinion.opinion_holder)
        except TypeError:
            pass
        
    for opinion in ko2.opinions:
        try:
            opinion_holders2.extend(opinion.opinion_holder)
        except TypeError:
            pass
    o1 = set(opinion_holders1)
    o2 = set(opinion_holders2)
    agr_a_b = len(o1.intersection(o2)) / max(len(o1), 0.000001)
    agr_b_a = len(o2.intersection(o1)) / max(len(o2), 0.000001)
    return (agr_a_b + agr_b_a) / 2

def create_sent_array(num_tokens):
    """Creates an array of zeros of the length of the tokens
       in the sentence."""
    
    return np.array(np.zeros((num_tokens,1)))

def create_opinion_array(sent_array, word_ids_and_opinions):
    """We are using the to identify the span
       the opinions and their opinion in the sentence array"""
    new = copy.deepcopy(sent_array)
    for w_ids, op in word_ids_and_opinions:
        new[w_ids-1] = opinion_dict[op]
        
    return new

def compare_opinion_expressions(KAF1, KAF2, error_function, type='kaf'):
    ko1 = KafNafOpinion(KAF1, type)
    ko2 = KafNafOpinion(KAF2, type)
    
    opinion_expressions1 = []
    opinion_expressions2 = []

    try:
        for opinion in ko1.opinions:
            opinion_expressions1.extend(opinion.opinion_expression)
        for opinion in ko2.opinions:
            opinion_expressions2.extend(opinion.opinion_expression)
    except:
        return None

    sent_array = create_sent_array(ko1.num_tokens)
    array1 = create_opinion_array(sent_array, opinion_expressions1)
    array2 = create_opinion_array(sent_array, opinion_expressions2)

    return error_function(array1, array2)


def mean_sqr_error(a, b):
    assert len(a) == len(b)
    n = len(a)
    return float(sum((a-b)**2))/n

opinion_dict = {'StrongNegative': 1,
      'Negative': 2,
      'Positive': 4,
      'StrongPositive': 5}

def print_stats(List1, DIR, type='kaf'):
    num_files = len(List1)
    num_tokens = get_number_tokens(List1, type)
    opinion_count1, opinion_targets1, opinion_holders1 = get_number_of_instances(List1, type)

    print('{0}'.format(DIR))
    print('Number of KAFs analyzed:  {0}'.format(num_files))
    print('Average length of review: {0:.1f}'.format(num_tokens/num_files))
    print()
    print('Opinion Count: {0}'.format(opinion_count1))
    print('Targets:       {0}'.format(opinion_targets1))
    print('Holders:       {0}'.format(opinion_holders1))
    print('-' * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default="corpora/ca")
    parser.add_argument('-type', default='kaf')
    args = parser.parse_args()

    kafs = open_kafs(args.dir)

    print_stats(kafs, args.dir, args.type)

if __name__ == "__main__":
    main()
