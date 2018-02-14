import numpy as np
import re
import os
import sys
import argparse

strength_dict = {'StrongPositive':5, 'Positive':4, 'Negative':2,
                 'StrongNegative':1}


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
            elif type == 'original':
                wid = re.findall('wid="w([0-9]*)"', tok)[0]
                word = re.findall('">(.*)</wf>', tok)[0]
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


    def get_opinion_tuple(self, kaf_tokens, opinion_holder,
                          opinion_target, opinion_expression):
        
        holder = None
        target = None
        
        if opinion_holder != None:
            holder = ' '.join([kaf_tokens[i] for i in opinion_holder])
        if opinion_target != None:
            target = ' '.join([kaf_tokens[i] for i in opinion_target])
        exp = [i for i,pol in opinion_expression]
        pol = [pol for i,pol in opinion_expression][0]
        exp = ' '.join([kaf_tokens[i] for i in exp])
        return (holder, target, exp, pol)


def get_kaf_opinions(kaf, type='kaf'):
    """
    Returns a dictionary of the opinions in the kaf
    where the keys are tuples of
    (opinion_holder, opinion_target, opinion_expression)
    and the values are the sentiment strength (1-4)
    """
    ko = KafNafOpinion(kaf, type=type)
    opinions = dict()
    for opinion in ko.opinions:
        if opinion.opinion_holder:
            holder_tokens = []
            for n in opinion.opinion_holder:
                holder_tokens.append(ko.tokens[n])
        else:
            holder_tokens = []

        if opinion.opinion_target:
            target_tokens = []
            for n in opinion.opinion_target:
                target_tokens.append(ko.tokens[n])
        else:
            target_tokens = []

        if opinion.opinion_expression:
            opinion_tokens = []
            opinion_strength = ''
            for n, strength in opinion.opinion_expression:
                opinion_strength = strength_dict[strength]
                opinion_tokens.append(ko.tokens[n])
        else:
            opinion_tokens = []
            opinion_strength = ''

        opinions[tuple(holder_tokens) + tuple(target_tokens) +
                 tuple(opinion_tokens)] = opinion_strength

    return opinions


def build_dataset(kafs, type='kaf'):
    dataset = dict()
    for i, kaf in enumerate(kafs):
        # print(i+1)
        dataset.update(get_kaf_opinions(kaf, type))
    return dataset


def train_dev_test(dataset):
    train_idx = int(len(dataset) * .7)
    dev_idx = int(len(dataset) * .8)
    train, dev, test = (dataset[:train_idx],
                        dataset[train_idx:dev_idx],
                        dataset[dev_idx:])
    return train, dev, test


def write_dataset(outfile, data):
    with open(outfile, 'w') as handle:
        for line in data:
            handle.write(line + '\n')


def create_opener_dataset(INDIR, OUTDIR, type='kaf'):
    fnames = os.listdir(INDIR)
    kafs = [open(os.path.join(INDIR, fname)).read() for fname in fnames]
    dataset = build_dataset(kafs, type)
    strn = [' '.join(s) for s, n in dataset.items() if n == 1]
    n = [' '.join(s) for s, n in dataset.items() if n == 2]
    p = [' '.join(s) for s, n in dataset.items() if n == 4]
    strp = [' '.join(s) for s, n in dataset.items() if n == 5]
    write_dataset(os.path.join(OUTDIR, 'strongneg.txt'), strn)
    write_dataset(os.path.join(OUTDIR, 'neg.txt'), n)
    write_dataset(os.path.join(OUTDIR, 'pos.txt'), p)
    write_dataset(os.path.join(OUTDIR, 'strongpos.txt'), strp)

    os.mkdir(OUTDIR + '/test/')
    os.mkdir(OUTDIR + '/train/')
    os.mkdir(OUTDIR + '/dev/')

    train, dev, test = train_dev_test(strn)
    write_dataset(OUTDIR + '/test/strongneg.txt', test)
    write_dataset(OUTDIR + '/dev/strongneg.txt', dev)
    write_dataset(OUTDIR + '/train/strongneg.txt', train)

    train, dev, test = train_dev_test(n)
    write_dataset(OUTDIR + '/test/neg.txt', test)
    write_dataset(OUTDIR + '/dev/neg.txt', dev)
    write_dataset(OUTDIR + '/train/neg.txt', train)

    train, dev, test = train_dev_test(p)
    write_dataset(OUTDIR + '/test/pos.txt', test)
    write_dataset(OUTDIR + '/dev/pos.txt', dev)
    write_dataset(OUTDIR + '/train/pos.txt', train)

    train, dev, test = train_dev_test(strp)
    write_dataset(OUTDIR + '/test/strongpos.txt', test)
    write_dataset(OUTDIR + '/dev/strongpos.txt', dev)
    write_dataset(OUTDIR + '/train/strongpos.txt', train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir')
    parser.add_argument('-outdir')
    parser.add_argument('-type', default='naf')

    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    create_opener_dataset(args.indir, args.outdir, type=args.type)

if __name__ == "__main__":
    main()

