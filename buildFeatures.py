import cPickle as pickle
from collections import Counter
import sys
from calculatePerplexityFeatures import calcTextNGramPerplexity


def buildNGramPerplexity(data,n=3):
    return calcTextNGramPerplexity(data, n)


def getErrorFeatures(data):
    return [getErrorFeaturePerArticle(d) for d in data]


def getErrorFeaturePerArticle(doc):
    bgrams = []
    for sen in doc:
        toks = sen.split()
        bgrams += zip(toks,toks[1:])
    filtered = [b for b in bgrams if b[0]==b[1]]
    if len(filtered) > 0:
        return 1
    else:
        return 0 


def getUnigramTypeTokenRatio(data):
    ret = []
    for article in data:
        ret.append(calculateUnigramTypeTokenRatio(article))
    return ret


def calculateUnigramTypeTokenRatio(doc):
    tokens = []
    for sen in doc:
        tokens = tokens + sen.split()
    types = Counter(tokens)
    tokenCount = len(tokens) - types[START_SYMBOL] - types[STOP_SYMBOL]
    typeCount = len(types) - 2 # -2 for start and stop symbol
    return float(typeCount)/(tokenCount)


def loadLabelledData(pklFile):
    with open(pklFile,'rb') as f:
        dat = pickle.load(f)
    return [dat['data'],dat['labels']]
