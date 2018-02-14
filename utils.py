from corpus_stats import *
import nltk

def get_opinions(kaf):
    targets = []
    for opinion in kaf.opinions:
        hol = []
        tar = []
        exp = []
        pol = []
        # assuming we have a holder
        try:
            for i in opinion.opinion_holder:
                hol.append(i)
        # if we don't, just pass
        except TypeError:
            pass

        # assuming we have a target
        try:
            for i in opinion.opinion_target:
                tar.append(i)
        # if we don't have a target
        except TypeError:
            pass

        # get opinion expression and polarity
        try:
            for j, p in opinion.opinion_expression:
                exp.append(j)
                pol.append(p)
        except TypeError:
            pass

        targets.append((hol, tar, exp, pol))
            
    return targets



def get_most_common(DIR):
    all_holders = []
    all_targets = []
    all_expressions = []
    
    for file in os.listdir(DIR):
        kaf = KafNafOpinion(open(os.path.join(DIR, file)).read(), 'naf')
        op = get_opinions(kaf)
        for hol, tar, exp, pol in op:
            try:
                if hol != []:
                    all_holders.append([kaf.tokens[i] for i in hol])
                if tar != []:
                    all_targets.append([kaf.tokens[i] for i in tar])
                exp = [kaf.tokens[i] for i in exp]
                all_expressions.append((exp, pol[0]))
            except:
                pass

    holder_list = nltk.FreqDist([w.lower() for s in all_holders for w in s])
    target_list = nltk.FreqDist([w.lower() for s in all_targets for w in s])
    expression_list = nltk.FreqDist([' '.join(w).lower() for w, s in all_expressions])

    return holder_list, target_list, expression_list

    
