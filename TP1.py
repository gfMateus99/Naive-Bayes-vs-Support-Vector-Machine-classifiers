# -*- coding: utf-8 -*-
"""
@author: Gonçalo Mateus, 53052
@author: Lourenço Vasconcelos, 52699
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
import math 

"""----------------------------------------------------"""
#  Preprocess: shuffle and standardize
"""----------------------------------------------------"""
# trainning data
mat_train = np.loadtxt('TP1_train.tsv', delimiter='\t')
data = shuffle(mat_train)
Y_r = data[:,4]
X_r = data[:,:4]
means = np.mean(X_r,axis=0)
stdevs = np.std(X_r,axis=0)
X_r = (X_r-means)/stdevs

# test data
mat_test = np.loadtxt('TP1_test.tsv', delimiter='\t')
Y_t = mat_test[:,4]
X_t = mat_test[:,:4]
X_t = (X_t-means)/stdevs

"""----------------------------------------------------"""
#  Partition training set into n folds
"""----------------------------------------------------"""
lr_graph_points = []
kde_graph_points = []
folds = 5
kf = StratifiedKFold(n_splits=folds)

"""----------------------------------------------------"""
#  Logistic Regression
"""----------------------------------------------------"""
def lr_calc_fold(X,Y, train_ix,valid_ix,C):
     """return error for train and validation sets"""
     reg = LogisticRegression(C=C, tol=1e-10)
     reg.fit(X[train_ix,:],Y[train_ix])
     prob = reg.predict_proba(X[:,:])[:,1]
     squares = (prob-Y)**2
     return np.mean(squares[train_ix]),np.mean(squares[valid_ix])
#  Cross validation of model to find the best C
best_c = 0
best_c_err = 10000
for i in range(-2, 13):
     c_value = 10**i
     tr_err = va_err = 0
     for tr_ix,va_ix in kf.split(Y_r,Y_r):
         r,v = lr_calc_fold(X_r,Y_r,tr_ix,va_ix, c_value)
         tr_err += r
         va_err += v
     if(round(va_err/folds, 5)<best_c_err):
         best_c = c_value
         best_c_err = va_err/folds
     lr_graph_points.append([math.log(c_value, 10), tr_err/folds, va_err/folds])
#  Train model with the best C value
reg = LogisticRegression(C=best_c, tol=1e-10)
reg.fit(X_r,Y_r)
test_error_lr = 1-reg.score(X_t,Y_t)
lr_errors = int(round(test_error_lr*Y_t.size))

"""----------------------------------------------------"""
#  Naïve Bayes classifier using KDE
"""----------------------------------------------------"""
def kernel_estimator(fit_data, train_data, valid_data, bandwidth):
     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
     kde.fit(fit_data)
     return kde.score_samples(train_data), kde.score_samples(valid_data)
def kde_calc_fold(X,Y,train_ix,valid_ix,bandwidth):
     """return error for train and validation sets"""
     f_train = X[train_ix[Y[train_ix]==0]]
     r_train = X[train_ix[Y[train_ix]==1]]
     c_train_0 = math.log(train_ix[Y[train_ix]==0].size/train_ix.size, math.e)
     c_train_1 = math.log(train_ix[Y[train_ix]==1].size/train_ix.size, math.e)
     c_valid_0 = c_train_0
     c_valid_1 = c_train_1
     for i in range(0,4):
        train,valid = kernel_estimator(f_train[:, [i]], X[train_ix][:, [i]], X[valid_ix][:, [i]], bandwidth)
        c_train_0 += train
        c_valid_0 += valid
        train,valid = kernel_estimator(r_train[:, [i]], X[train_ix][:, [i]], X[valid_ix][:, [i]], bandwidth)
        c_train_1 += train
        c_valid_1 += valid
     t_predict = np.argmax([c_train_0, c_train_1], axis = 0) == Y[train_ix]
     train_error = (train_ix.size-np.sum(t_predict))/train_ix.size
     v_predict = np.argmax([c_valid_0, c_valid_1], axis = 0) == Y[valid_ix]
     valid_error = (valid_ix.size-np.sum(v_predict))/valid_ix.size
     return train_error, valid_error
def kernel_test(fit_data, test_data, bandwidth):
     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
     kde.fit(fit_data)
     return kde.score_samples(test_data)
def calc_kde_true_error(X_r,Y_r, X_t, Y_t, bandwidth):
     """return error for train and validation sets"""
     f_train = X_r[Y_r==0]
     r_train = X_r[Y_r==1]
     c_test_0 = math.log(Y_r[Y_r==0].size/Y_r.size, math.e)
     c_test_1 = math.log(Y_r[Y_r==1].size/Y_r.size, math.e)
     for i in range(0,4):
        c_test_0 += kernel_test(f_train[:, [i]], X_t[:, [i]], bandwidth)
        c_test_1 += kernel_test(r_train[:, [i]], X_t[:, [i]], bandwidth)
     predicts = np.argmax([c_test_0, c_test_1], axis = 0)
     true_predict = predicts == Y_t
     number_errors = Y_t.size-np.sum(true_predict)
     true_error = (number_errors)/Y_t.size
     return true_error, number_errors, predicts
 
#  Cross validation of model to find the best bandwidth
best_bandwidth = 0
best_bandwidth_error = 1
for i in range(1, 31):
    bandwidth = i*0.02
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = kde_calc_fold(X_r,Y_r,tr_ix,va_ix, bandwidth)
        tr_err += r
        va_err += v
    if(va_err/folds<best_bandwidth_error):
        best_bandwidth=bandwidth
        best_bandwidth_error = va_err/folds
    kde_graph_points.append([bandwidth, tr_err/folds, va_err/folds])
#  Train model with the best bandwidth value
test_error_kde, kde_errors, kde_pred_Ys = calc_kde_true_error(X_r, Y_r, X_t, Y_t, best_bandwidth)

"""----------------------------------------------------"""
#  Gaussian Naive Bayes classifier
"""----------------------------------------------------"""
clf = GaussianNB()
clf.fit(X_r, Y_r)
test_error_gbn = 1 - clf.score(X_t, Y_t)
gbn_errors = int(round(test_error_gbn*Y_t.size))

"""----------------------------------------------------"""
#   Plot the training and cross-validation error as a function of the logarithm of C
"""----------------------------------------------------"""
p = np.array(lr_graph_points)
plt.title("Training and cross-validation errors for the C parameter")
plt.plot(p[:,0], p[:,1], '-', color='blue', label="Training error")
plt.plot(p[:,0], p[:,2], '-', color='red', label="Cross-Validation error")
plt.legend()
plt.savefig('LR.png', dpi=300)
plt.show()
plt.close()
p = np.array(kde_graph_points)
plt.title("Training and cross-validation errors for the bandwidth parameter")
plt.plot(p[:,0], p[:,1], '-', color='blue', label="Training error")
plt.plot(p[:,0], p[:,2], '-', color='red', label="Cross-Validation error")
plt.legend()
plt.savefig('NB.png', dpi=300)
plt.show()
plt.close()

"""----------------------------------------------------"""
#  Errors and classifiers
"""----------------------------------------------------"""
def normal_test_calc(test_size, errors):
    delta  = np.sqrt(test_size*(errors/test_size)*(1-(errors/test_size)))
    return 1.96*delta
print('---------------- Best Parameters -----------------')
print('Best C : ', best_c)
print('Best Bandwidth : ', best_bandwidth)
print('------------------- True Error -------------------')
print('Logistic Error: ', test_error_lr)
print('Naïve Bayes Error: ', test_error_kde)
print('Gaussian Naïve Bayes Error :', test_error_gbn)
print('--------------- Normal Classifier ----------------')
reg_dev = normal_test_calc(Y_t.size, lr_errors)
print('Logistic Regression: ', lr_errors, ' +/- ', reg_dev)
kde_dev = normal_test_calc(Y_t.size, kde_errors)
print('Naïve Bayes: ', kde_errors, ' +/- ', kde_dev)
gnb_dev = normal_test_calc(Y_t.size, gbn_errors)
print('Gaussian Naïve Bayes: ', gbn_errors, ' +/- ', gnb_dev)
print("---------------- McNemar's test ------------------")

reg_pred_Ys = reg.predict(X_t)
gnb_pred_Ys = clf.predict(X_t)

reg_vs_kde = [0,0]
reg_vs_gnb = [0,0]
kde_vs_gnb = [0,0]

i=0
for x in Y_t:
    if(reg_pred_Ys[i]==x and kde_pred_Ys[i]!=x):
        reg_vs_kde[0]+= 1
    if(reg_pred_Ys[i]!=x and kde_pred_Ys[i]==x):
        reg_vs_kde[1]+=1
    if(reg_pred_Ys[i]==x and gnb_pred_Ys[i]!=x):
        reg_vs_gnb[0]+=1
    if(reg_pred_Ys[i]!=x and gnb_pred_Ys[i]==x):
        reg_vs_gnb[1]+=1
    if(kde_pred_Ys[i]==x and gnb_pred_Ys[i]!=x):
        kde_vs_gnb[0]+=1
    if(kde_pred_Ys[i]!=x and gnb_pred_Ys[i]==x):
        kde_vs_gnb[1]+=1
    i+=1

reg_kde = ((np.absolute(reg_vs_kde[1]-reg_vs_kde[0])-1)**2)/(reg_vs_kde[1]+reg_vs_kde[0])
reg_gnb = ((np.absolute(reg_vs_gnb[1]-reg_vs_gnb[0])-1)**2)/(reg_vs_gnb[1]+reg_vs_gnb[0])
kde_gnb = ((np.absolute(kde_vs_gnb[1]-kde_vs_gnb[0])-1)**2)/(kde_vs_gnb[1]+kde_vs_gnb[0])

print('Logistic Regression vs Naïve Bayes : ', reg_kde)
print('Logistic Regression vs Gaussian Naïve Bayes test: ', reg_gnb)
print('Naïve Bayes vs Gaussian Naïve Bayes test: ', kde_gnb)