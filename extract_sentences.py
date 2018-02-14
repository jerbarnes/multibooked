import sys
import os
import re
from extract_opinions import *

def get_sents(kaf, type='kaf'):
    token_section = get_token_section(kaf)
    num_sents = max([int(w) for w in re.findall('sent="([0-9]*)"', kaf)])
    sents = {}
    for i in range(1, num_sents+1):
        sents[i] = {}
        sents[i]['tokens'] = {}
        for line in token_section.split('<wf'):
            try:
                if int(re.findall('sent="([0-9]*)"', line)[0]) == i:
                    if type =='kaf':
                        wid = re.findall('wid="w_([0-9]*)"', line)[0]
                        word = re.findall('sent="[0-9]*">(.*)</wf>', line)[0]
                    elif type == 'original':
                        wid = re.findall('wid="w([0-9]*)"', line)[0]
                        word = re.findall('">(.*)</wf>', line)[0]
                    else:
                        wid = re.findall('id="w([0-9]*)"', line)[0]
                        word = re.findall('para="[0-9]*">(.*)</wf>', line)[0]
                    sents[i]['tokens'][int(wid)] = word
            except:
                pass
    return sents

def get_sent_opinions(kaf, type='kaf'):
    opinion_section = get_opinion_section(kaf)
    raw_ops = get_opinions(opinion_section)

    opinions = {}
    for op in raw_ops:
        exp = get_opinion_expression(op, type=type)
        opinions.update(exp)
    return opinions

def map_kaf_annotations_to_sentences(kaf, type='kaf'):
    sents = get_sents(kaf, type=type)
    opinions = get_sent_opinions(kaf, type=type)
    for sent in sents.values():
        sent['opinions'] = {}
        for widx in sent['tokens'].keys():
            try:
                op = opinions[widx]
                sent['opinions'][widx] = op
            except KeyError:
                pass

    return sents

def mapped_sentences(kafs, type='kaf'):

    dataset = {}
    
    for i, kaf in enumerate(kafs):
        try:
            sents = map_kaf_annotations_to_sentences(kaf, type=type)
            for sent in sents.values():
                opinions = sent['opinions'].values()
                sent = ' '.join([sent['tokens'][i] for i in sorted(sent['tokens'])])
                if len(set(opinions)) == 1:
                    label = list(opinions)[0]
                    dataset[sent] = label
                elif set(opinions) == {'Positive', 'StrongPositive'}:
                    label = 'Positive'
                    dataset[sent] = label
                elif set(opinions) == {'Negative', 'StrongNegative'}:
                    label = 'Negative'
                    dataset[sent] = label
                elif len(set(opinions)) > 1:
                    label = 'Mixed'
                    dataset[sent] = label
        except:
            print("unable to process kaf %i"%i)

    return dataset

def mapped_documents(kafs, type='kaf'):
    dataset = {}
    for i, kaf in enumerate(kafs):
        try:
            sents = map_kaf_annotations_to_sentences(kaf, type=type)
            review = ' '.join([' '.join([sents[k]['tokens'][i] for i in sorted(sents[k]['tokens'].keys())]) for k in sorted(sents.keys())])
            opinions = []
            for sent in sents.values():
                for opinion in sent['opinions'].values():
                    opinions.append(opinion)
            label = max(set(opinions), key=opinions.count)
            dataset[review] = label
        except:
            print('unable to process kaf %i' %i)
    return dataset


def create_mapped_dataset(INDIR, OUTDIR, type='kaf', level='sents'):
    fnames = os.listdir(INDIR)
    kafs = [open(os.path.join(INDIR, fname)).read() for fname in fnames]
    if level == 'sents':
        dataset = mapped_sentences(kafs, type)
    else:
        dataset = mapped_documents(kafs, type)
    strn = [s for s,n in dataset.items() if n == 'StrongNegative']
    n = [s for s,n in dataset.items() if n == 'Negative']
    p = [s for s,n in dataset.items() if n == 'Positive']
    strp = [s for s,n in dataset.items() if n == 'StrongPositive']
    mixed = [s for s,n in dataset.items() if n == 'Mixed']
    write_dataset(os.path.join(OUTDIR, 'strongneg.txt'), strn)
    write_dataset(os.path.join(OUTDIR, 'neg.txt'), n)
    write_dataset(os.path.join(OUTDIR, 'pos.txt'), p)
    write_dataset(os.path.join(OUTDIR, 'strongpos.txt'), strp)
    write_dataset(os.path.join(OUTDIR, 'mixed.txt'), mixed)

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

    train, dev, test = train_dev_test(mixed)
    write_dataset(OUTDIR + '/test/mixed.txt', test)
    write_dataset(OUTDIR + '/dev/mixed.txt', dev)
    write_dataset(OUTDIR + '/train/mixed.txt', train)

def train_dev_test(dataset):
    train_idx = int(len(dataset) * .7)
    dev_idx = int(len(dataset) * .8)
    train, dev, test = dataset[:train_idx], dataset[train_idx:dev_idx], dataset[dev_idx:]
    return train, dev, test

def write_dataset(outfile, data):
    with open(outfile, 'w') as handle:
        for line in data:
            handle.write(line+'\n')

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir')
    parser.add_argument('-outdir')
    parser.add_argument('-type', default='naf')
    parser.add_argument('-level', default='sentences', help='either sentences or documents')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
    	os.mkdir(args.outdir)

    create_mapped_dataset(args.indir, args.outdir, type=args.type, level=args.level)


if __name__ == "__main__":

    args = sys.argv
    main(args)
