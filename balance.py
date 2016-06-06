import cPickle as pickle
from experiment import load_data
from cPickle import HIGHEST_PROTOCOL


def avg_len(articles):
    total_sents = 0.0
    total_articles = len(articles)
    for article in articles:
        total_sents += len(article)
    print total_articles, total_sents
    return total_sents/total_articles


def get_article_idx(articles, l):
    min_idx, min_len = -1, float('inf')
    for i in xrange(len(articles)):
        article = articles[i]
        art_len = len(article)
        if art_len >= l:
            if art_len < min_len:
                min_idx = i
                min_len = art_len
    return min_idx


def get_articles_from_label(articles, labels, l):
    return [pair[0] for pair in zip(articles, labels) \
            if pair[1] == l]


def balance_dataset(articles):
    bal_data = []
    lengths = [1, 1, 2, 3, 4, 5, 7, 10, 15, 20]
    iter = 0
    while True:
        for l in lengths:
            art_idx = get_article_idx(articles, l)
            if art_idx == -1:
                return bal_data
            new_article = articles[art_idx][:l]
            rem_article = articles[art_idx][l:]
            bal_data.append(new_article)
            del articles[art_idx]
            articles.append(rem_article)
    return bal_data


def generate_balanced_dataset():
    train_file = '../data_pickle/training.pkl'
    test_file = '../data_pickle/dev.pkl'
    train = load_data(train_file)
    test = load_data(test_file)
    
    real_articles = get_articles_from_label(train['data'], train['labels'], 1)
    fake_articles = get_articles_from_label(train['data'], train['labels'], 0)
    
    bal_real_articles = balance_dataset(real_articles)
    bal_fake_articles = balance_dataset(fake_articles)
    

    new_articles = bal_real_articles + bal_fake_articles
    new_labels = ([1] * len(bal_real_articles)) + ([0] * len(bal_fake_articles))
    bal_data = { 'data':new_articles, 'labels':new_labels }
