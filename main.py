import sys
from constants import *
from predict import *
from calculatePerplexityFeatures import calcTextNGramPerplexity


def read_data(file):
    articles = [] 
    lines = file.readlines()
    article = None
    for i in range(len(lines)):
        if lines[i].strip() == ARTICLE_DELIMITER:
            if article is not None:
                articles.append(article)
            article = []
        elif len(lines[i].strip()) == 0:
            continue
        else:
            article.append(lines[i].strip())
    if article is not None:
        articles.append(article)
    return articles


def get_vals(filename, type='float'):
    with open(filename,'r') as f:
        lines = f.readlines()
    if type == 'float':
        return [float(line.strip('\n')) for line in lines]
    else:
        return [int(line.strip('\n')) for line in lines]


def get_train_grams(): 
    train_files = [TRAIN_3G, TRAIN_4G, TRAIN_5G, TRAIN_6G, TRAIN_7G]
    train_perps = []
    for train_file in train_files:
        train_perps.append(get_vals(train_file))
    return train_perps


def get_test_grams(articles):
    test_ns = [3, 4, 5, 6, 7]
    test_perps = []
    for n in test_ns:
        test_perps.append(calcTextNGramPerplexity(articles, n))
    return test_perps


def main():
    # read test file
    test_articles = read_data(sys.stdin)
    train_grams = get_train_grams()
    test_grams = get_test_grams(test_articles)
    train_labels = get_vals(TRAIN_LABELS, 'int')
    test_labels, test_probs = predict_labels(train_grams, train_labels, test_grams)
    for i in xrange(len(test_labels)):
        print '%.8f %.8f %d' % (test_probs[i][0], test_probs[i][1], test_labels[i])


if __name__ == '__main__':
    main()