# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:22:23 2020

@author: gaura
"""

# importing files and libraries
import pandas as pd
import numpy as np
dataset_train = pd.read_excel('Training_set.xlsx')
dataset_test = pd.read_excel('Test_set.xlsx')


# data pre - processing
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

def train_corpus(dataset, value):
    corpus_t = []
    list_corpus_t = []
    for i in range(0, value):
        j = [dataset['News'][i]]
        corpus_t.append(j)
        j = ' '.join(j)
        list_corpus_t.append(j)
    return corpus_t, list_corpus_t
        
def t_corpus(dataset, value):
    corpus_t = []
    list_corpus_t = []
    for i in range(0, value):
        j = re.sub('[^a-zA-Z]', ' ', dataset['News'][i])
        j = j.lower()
        j = j.split()
        ps = PorterStemmer()
        j = [ps.stem(word) for word in j if not word in set(stopwords.words('english'))]
        corpus_t.append(j)
        j = ' '.join(j)
        list_corpus_t.append(j)
    return corpus_t, list_corpus_t


# train and test data pre - processing
corpus_train, list_corpus_train = t_corpus(dataset_train, 1247)
corpus_test, list_corpus_test = t_corpus(dataset_test, 500)


# defining y_train and y_test
y_train = dataset_train.iloc[:, 1].values
y_test = dataset_test.iloc[:, 1].values


# count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

def count_vec(train_data, test_data):
    cv = CountVectorizer(max_features = 500)
    X_train = cv.fit_transform(train_data).toarray()
    X_test = cv.transform(test_data).toarray()
    return X_train, X_test, cv

X_train_cv, X_test_cv, cv = count_vec(list_corpus_train, list_corpus_test)


# tf - idf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vec(train_data, test_data):
    
    tv = TfidfVectorizer(max_features = 500)
    X_train = tv.fit_transform(train_data).toarray()
    X_test = tv.transform(test_data).toarray()
    return X_train, X_test, tv

X_train_tv, X_test_tv, tv = tfidf_vec(list_corpus_train, list_corpus_test)


# word2vec
import gensim
word2vec_path = "GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#word2vec.save("word2vec_model_file.model")
#from gensim.models import Word2Vec
#model = Word2Vec.load("word2vec_model_file.model")

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, corpus):
    embeddings = map(lambda x: get_average_word2vec(x, vectors), corpus)
    return list(embeddings)

X_train_w2v = get_word2vec_embeddings(word2vec, list_corpus_train)
X_test_w2v = get_word2vec_embeddings(word2vec, list_corpus_test)


# latent dirichlet allocation

from sklearn.decomposition import LatentDirichletAllocation

def lda_(train_data, test_data):
    
    lda = LatentDirichletAllocation(n_components = 10)
    X_train = lda.fit_transform(train_data)
    X_test = lda.transform(test_data)
    return X_train, X_test

X_train_cv_lda, X_test_cv_lda = lda_(X_train_cv, X_test_cv)
X_train_tv_lda, X_test_tv_lda = lda_(X_train_tv, X_test_tv)


# latent semantic analysis

from sklearn.decomposition import TruncatedSVD

def lsa_(train_data, test_data):
    
    tsvd = TruncatedSVD(n_components = 100)
    X_train = tsvd.fit_transform(train_data)
    X_test = tsvd.transform(test_data)
    return X_train, X_test

X_train_cv_lsa, X_test_cv_lsa = lsa_(X_train_cv, X_test_cv)
X_train_tv_lsa, X_test_tv_lsa = lsa_(X_train_tv, X_test_tv)


# principal component analysis

from sklearn.decomposition import PCA

def pca_(train_data, test_data):
    
    pca = PCA(n_components = 100)
    X_train = pca.fit_transform(train_data)
    X_test = pca.transform(test_data)
    return X_train, X_test

X_train_cv_pca, X_test_cv_pca = pca_(X_train_cv, X_test_cv)
X_train_tv_pca, X_test_tv_pca = pca_(X_train_tv, X_test_tv)


# scaling data between [0, 1] (only required in naive bayes)

from sklearn.preprocessing import MinMaxScaler

def minmax_scaler(train_data, test_data):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data)
    X_test = scaler.transform(test_data)
    return X_train, X_test

X_train_w2v_scal, X_test_w2v_scal = minmax_scaler(X_train_w2v, X_test_w2v)
#X_train_tv_lsa_scal, X_test_tv_lsa_scal = standard_scaler(X_train_tv_lsa, X_test_tv_lsa)


# naive bayes

from sklearn.naive_bayes import MultinomialNB

def bayes_naive(X_train, X_test):
    
    classifier_mnb = MultinomialNB()
    classifier_mnb.fit(X_train, y_train)
    y_pred = classifier_mnb.predict(X_test)
    return y_pred, classifier_mnb


# naive bayes with count vectorizer, tf-idf, word2vec

y_pred_mnb_cv, classifier_mnb_cv = bayes_naive(X_train_cv_lda, X_test_cv_lda)
y_pred_mnb_tv, classifier_mnb_tv = bayes_naive(X_train_tv_lda, X_test_tv_lda)
y_pred_mnb_w2v, classifier_mnb_w2v = bayes_naive(X_train_w2v_scal, X_test_w2v_scal)


# random forest classifier

from sklearn.ensemble import RandomForestClassifier

def rf_classifier(X_train, X_test):
    classifier_rfc = RandomForestClassifier(criterion = 'entropy', random_state = 0)
    classifier_rfc.fit(X_train, y_train)
    y_pred = classifier_rfc.predict(X_test)
    return y_pred, classifier_rfc


# random forest with countvectorizer, tf-idf, word2vec

y_pred_rfc_cv, classifier_rfc_cv = rf_classifier(X_train_cv_lsa, X_test_cv_lsa)
y_pred_rfc_tv, classifier_rfc_tv = rf_classifier(X_train_tv_lsa, X_test_tv_lsa)
y_pred_rfc_w2v, classifier_rfc_w2v = rf_classifier(X_train_w2v, X_test_w2v)


# decision tree classifier

from sklearn.tree import DecisionTreeClassifier

def dt_classifier(X_train, X_test):
    
    classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_dtc.fit(X_train, y_train)
    y_pred = classifier_dtc.predict(X_test)
    return y_pred, classifier_dtc


# decision tree with count vectorizer, tf-idf, word2vec

y_pred_dtc_cv, classifier_dtc_cv = dt_classifier(X_train_cv_lsa, X_test_cv_lsa)
y_pred_dtc_tv, classifier_dtc_tv = dt_classifier(X_train_tv_lsa, X_test_tv_lsa)
y_pred_dtc_w2v, classifier_dtc_w2v = dt_classifier(X_train_w2v, X_test_w2v)


# confusion matrix
from sklearn.metrics import confusion_matrix
def confus_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm


# confusion matrix for naive bayes with count vectorizer, tf-idf, word2vec

mnb_cv = confus_matrix(y_test, y_pred_mnb_cv)
mnb_tv = confus_matrix(y_test, y_pred_mnb_tv)
mnb_w2v = confus_matrix(y_test, y_pred_mnb_w2v)


# confusion matrix for random forest with count vectorizer, tf-idf, word2vec

rfc_cv = confus_matrix(y_test, y_pred_rfc_cv)
rfc_tv = confus_matrix(y_test, y_pred_rfc_tv)
rfc_w2v = confus_matrix(y_test, y_pred_rfc_w2v)


#confusion matrix for decision tree with count vectorizer, tf-idf, word2vec

dtc_cv = confus_matrix(y_test, y_pred_dtc_cv)
dtc_tv = confus_matrix(y_test, y_pred_dtc_tv)
dtc_w2v = confus_matrix(y_test, y_pred_dtc_w2v)


# checking important features

def features(vectorizer, model, n = 5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes


# features in naive bayes with count vectorizer

important_mnb_cv = features(cv, classifier_mnb_cv, 20)


# features in naive bayes with tf - idf

important_mnb_tv = features(tv, classifier_mnb_tv, 30)