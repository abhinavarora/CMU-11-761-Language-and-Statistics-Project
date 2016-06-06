import numpy as np
from  math import log
from main import get_vals

from sklearn import linear_model, datasets
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing


def build_features(grams):
    nrows = len(grams[0])
    ncols = len(grams)
    X = np.zeros((nrows, ncols))
    for j in xrange(ncols):
        for i in xrange(nrows):
            X[i, j] = -log(grams[j][i], 2)
    return  X


def predict_labels(train_grams, train_labels, test_grams):
    X_train = build_features(train_grams)
    Y_train = train_labels
    X_test = build_features(test_grams)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    model = svm.SVC(C=10, kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    Y_test = model.predict(X_test)
    Y_test_probs = model.predict_proba(X_test)
    #Y_test_actual = get_vals('developmentSetLabels.dat', 'int')
    #print sum(Y_test_actual == Y_test)/float(len(Y_test))
    return Y_test, Y_test_probs

