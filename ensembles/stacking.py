# Python script for kaggle house price challenge

# import functions
import numpy as np
import pandas as pd
pd.set_option('display.precision',20)
from scipy import stats
from scipy import optimize

from sklearn import linear_model, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost.training import train
from xgboost.sklearn import XGBClassifier


import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# def to compare goodness of fit on training set
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Preprocessing on the data set has already been done in R

# these files have all features
train_1_9_df = pd.read_csv("train-1-9.csv")
test_1_9_df = pd.read_csv("test-1-9.csv")

# these files have all features
train_1_7_df = pd.read_csv("train-1-7.csv")
test_1_7_df = pd.read_csv("test-1-7.csv")

# these files have all features
train_1_3_df = pd.read_csv("train-1-3.csv")
test_1_3_df = pd.read_csv("test-1-3.csv")

train_1_3_dropped_df = pd.read_csv("train-1-3-dropped.csv")
test_1_3_dropped_df = pd.read_csv("test-1-3-dropped.csv")

train_1_2_df = pd.read_csv("train-1-2.csv")
test_1_2_df = pd.read_csv("test-1-2.csv")

# Legacy files with all features as of 12/31/16
train_12_31_df = pd.read_csv("train-12-31.csv")
test_12_31_df = pd.read_csv("test-12-31.csv")

# these have Amit Choudhary's features (there is code below to deal with some differences)
label_AC_df = pd.read_csv("AC_label.csv")
train_AC_df = pd.read_csv("AC_train.csv")
test_AC_df = pd.read_csv("AC_test.csv")

# low variance columns have been dropped in hyperparameter-search.py
# these files have dropped cat features with less than 10 nonzero entries
train_12_31_dropped_df = pd.read_csv("train-12-31-dropped.csv")
test_12_31_dropped_df = pd.read_csv("test-12-31-dropped.csv")


# Set up predictors and response
y_train_1_9 = train_1_9_df['LogSalePrice'].values
x_train_1_9 = train_1_9_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_1_9 = test_1_9_df.drop(['Id'],axis=1).values

y_train_1_7 = train_1_7_df['LogSalePrice'].values
x_train_1_7 = train_1_7_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_1_7 = test_1_7_df.drop(['Id'],axis=1).values

y_train_1_3 = train_1_3_df['LogSalePrice'].values
x_train_1_3 = train_1_3_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_1_3 = test_1_3_df.drop(['Id'],axis=1).values

y_train_1_3_dropped = train_1_3_dropped_df['LogSalePrice'].values
x_train_1_3_dropped = train_1_3_dropped_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_1_3_dropped = test_1_3_dropped_df.drop(['Id'],axis=1).values

y_train_1_2 = train_1_2_df['LogSalePrice'].values
x_train_1_2 = train_1_2_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_1_2 = test_1_2_df.drop(['Id'],axis=1).values


y_train_12_31 = train_12_31_df['LogSalePrice'].values
x_train_12_31 = train_12_31_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_12_31 = test_12_31_df.drop(['Id'],axis=1).values

y_train_12_31_dropped = train_12_31_dropped_df['LogSalePrice'].values
x_train_12_31_dropped = train_12_31_dropped_df.drop(['Id','LogSalePrice'],axis=1).values
x_test_12_31_dropped = test_12_31_dropped_df.drop(['Id'],axis=1).values

# AC's features are a bit different
y_train_AC = label_AC_df['SalePrice'].values
x_train_AC = train_AC_df.drop(['Id','_RoofMatl_ClyTile'],axis=1).values
x_test_AC = test_AC_df.drop(['Id'],axis=1).values

print("1-9 features training set size:", x_train_1_9.shape)
print("1-9 features test set size:", x_test_1_9.shape)

print("1-7 features training set size:", x_train_1_7.shape)
print("1-7 features test set size:", x_test_1_7.shape)

print("1-3 features training set size:", x_train_1_3.shape)
print("1-3 features test set size:", x_test_1_3.shape)

print("1-3 w dropfeatures training set size:", x_train_1_3_dropped.shape)
print("1-3 w drop features test set size:", x_test_1_3_dropped.shape)

print("1-2 features training set size:", x_train_1_2.shape)
print("1-2 features test set size:", x_test_1_2.shape)

print("12-31 features training set size:", x_train_12_31.shape)
print("12-31 features test set size:", x_test_12_31.shape)
print("12-31 w drop features training set size:", x_train_12_31_dropped.shape)
print("12-31 w drop features test set size:", x_test_12_31_dropped.shape)

print("AC features training set size:", x_train_AC.shape)
print("AC features training response set size:", y_train_AC.shape)
print("AC features test set size:", x_test_AC.shape)

y_train = y_train_1_9
x_train = [x_train_1_9, x_train_1_7, x_train_1_3, x_train_1_3_dropped, x_train_1_2,
           x_train_12_31, x_train_12_31_dropped, x_train_AC]
x_test = [x_test_1_9, x_test_1_7, x_test_1_3, x_test_1_3_dropped, x_test_1_2,
          x_test_12_31, x_test_12_31_dropped, x_test_AC]
# x_train = [x_train_1_9, x_train_1_7, x_train_AC]
# x_test = [x_test_1_9, x_test_1_7, x_test_AC]

# Cross-validation

kfold = KFold(n_splits=10, random_state=7)

# models
rms_1_9 = [linear_model.Ridge(alpha = 24),
           linear_model.Lasso(alpha=0.0004, max_iter=50000),
           linear_model.LassoLars(alpha=0.00012, max_iter=50000),
           linear_model.ElasticNet(alpha = 0.0009, l1_ratio=0.54, max_iter=15000, random_state=7),
           RandomForestRegressor(n_estimators = 1500, max_depth = 16, random_state = 7),
           svm.SVR(C=2, cache_size=200, coef0=0.0, degree=3, epsilon=0.036, gamma=0.0009, kernel='rbf',
                   max_iter=-1, shrinking=True, tol=0.001, verbose=False),
           xgb.XGBRegressor(max_depth=2, min_child_weight=1.1, gamma=0, subsample=1, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.2, learning_rate=0.06, n_estimators=900, seed=42,
                            nthread=-1, silent=1),
           linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=33)
           ]

rms_1_7 = [linear_model.Ridge(alpha = 14),
          linear_model.Lasso(alpha=0.0005, max_iter=50000),
          linear_model.LassoLars(alpha=0.00011, max_iter=50000),
          linear_model.ElasticNet(alpha = 0.0009, l1_ratio=0.54, max_iter=15000, random_state=7),
          RandomForestRegressor(n_estimators = 600, max_depth = 21, random_state = 7),
          svm.SVR(C=25.2, cache_size=200, coef0=0.0, degree=3, epsilon=0.0037, gamma=0.00006, kernel='rbf',
                  max_iter=-1, shrinking=True, tol=0.001, verbose=False),
          xgb.XGBRegressor(max_depth = 3, min_child_weight = 3, gamma = 0, subsample = 0.92, colsample_bytree = 0.6,
                           reg_alpha = 0.2, reg_lambda = 0.2, learning_rate = 0.1, n_estimators = 1100, seed = 42,
                           nthread = -1, silent = 1),
          linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=28)
          ]

rms_1_3 = [linear_model.Ridge(alpha = 26),
           linear_model.Lasso(alpha=0.00068, max_iter=50000),
           linear_model.LassoLars(alpha=0.00012, max_iter=50000),
           linear_model.ElasticNet(alpha = 0.0014, l1_ratio=0.49, max_iter=15000, random_state=7),
           RandomForestRegressor(n_estimators = 2300, max_depth = 19, random_state = 7),
           svm.SVR(C=1.95, cache_size=200, coef0=0.0, degree=3, epsilon=0.0167, gamma='auto',kernel='rbf',
                   max_iter=-1, shrinking=True, tol=0.001, verbose=False),
           xgb.XGBRegressor(max_depth = 2, min_child_weight = 3.1, gamma = 0.015, subsample = 0.81,
                            colsample_bytree = 0.80, reg_alpha = 0.19, reg_lambda = 0.2, learning_rate = 0.071,
                            n_estimators = 3400, seed = 42, nthread = -1, silent = 1),
           linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=37)
           ]

rms_1_3_dropped = [linear_model.Ridge(alpha = 2.6),
                   linear_model.Lasso(alpha=0.000151, max_iter=50000),
                   linear_model.LassoLars(alpha=0.00014, max_iter=50000),
                   linear_model.ElasticNet(alpha = 0.0002, l1_ratio=0.00015, max_iter=15000, random_state=7),
                   RandomForestRegressor(n_estimators = 600, max_depth = 19, random_state = 7),
                   svm.SVR(C=31.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.02535, gamma=0.000033,
                           kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
                   xgb.XGBRegressor( max_depth = 3, min_child_weight = 3.1, gamma = 0.018, subsample = 0.8,
                                     colsample_bytree = 0.8, reg_alpha = 0.2, reg_lambda = 0.2, learning_rate = 0.07,
                                     n_estimators = 900, seed = 42, nthread = -1, silent = 1),
                   linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=80)
                   ]

rms_1_2 = [linear_model.Ridge(alpha = 19.2),
           linear_model.Lasso(alpha=0.000838, max_iter=50000),
           linear_model.LassoLars(alpha=0.000181, max_iter=50000),
           linear_model.ElasticNet(alpha = 0.00142, l1_ratio=0.501, max_iter=15000, random_state=7),
           RandomForestRegressor(n_estimators = 620, max_depth = 15, random_state = 7),
           svm.SVR(C=1.56, cache_size=200, coef0=0.0, degree=3, epsilon=0.00735, gamma='auto',
                   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
           xgb.XGBRegressor( max_depth = 2, min_child_weight = 3.1, gamma = 0.00968, subsample = 0.778,
                             colsample_bytree = 0.83, reg_alpha = 0.2085, reg_lambda = 0.1991,
                             learning_rate = 0.05, n_estimators = 2000, seed = 42, nthread = -1,
                             silent = 1),
           linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=28)
           ]

rms_12_31 = [linear_model.Ridge(alpha = 74.682),
             linear_model.Lasso(alpha=0.000866, max_iter=50000),
             linear_model.LassoLars(alpha=0.000194, max_iter=50000),
             linear_model.ElasticNet(alpha = 0.001735, l1_ratio=0.5, max_iter=15000, random_state=7),
             RandomForestRegressor(n_estimators=570, max_depth=26, random_state=7),
             svm.SVR(C=1.15, cache_size=200, coef0=0.0, degree=3, epsilon=0.0222, gamma='auto', kernel='rbf',
                     max_iter=-1, shrinking=True, tol=0.001, verbose=False),
             xgb.XGBRegressor(max_depth = 3, min_child_weight = 1.1, gamma = 0.0529, subsample = 0.8,
                              colsample_bytree = 0.202, reg_alpha = 0.152, reg_lambda = 0.19705,
                              learning_rate = 0.2, n_estimators = 13000, seed = 42, nthread = -1, silent = 1),
             linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=27)
             ]


rms_12_31_dropped = [linear_model.Ridge(alpha = 78.17),
                     linear_model.Lasso(alpha=0.0007753, max_iter=50000),
                     linear_model.LassoLars(alpha=0.000122, max_iter=50000),
                     linear_model.ElasticNet(alpha = 0.001468, l1_ratio=0.5, max_iter=15000, random_state=7),
                     RandomForestRegressor(n_estimators=620, max_depth=15, random_state=7),
                     svm.SVR(C=1.2, cache_size=200, coef0=0.0, degree=3, epsilon=0.0211, gamma='auto', kernel='rbf',
                             max_iter=-1, shrinking=True, tol=0.001, verbose=False),
                     xgb.XGBRegressor(max_depth = 3, min_child_weight = 2.02,gamma = 0, subsample = 0.81985,
                                      colsample_bytree = 0.9, reg_alpha = 0.193, reg_lambda = 0.200481,
                                      learning_rate = 0.0141, n_estimators = 3500, seed = 42, nthread = -1,
                                      silent = 1),
                     linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=36)
                     ]

rms_AC = [linear_model.Ridge(alpha = 58.28),
          linear_model.Lasso(alpha=0.000378, max_iter=50000),
          linear_model.LassoLars(alpha=0.0005971, max_iter=50000),
          linear_model.ElasticNet(alpha = 0.000739, l1_ratio=00.5017, max_iter=15000, random_state=7),
          RandomForestRegressor(n_estimators=780, max_depth=22, random_state=7),
          svm.SVR(C=0.82, cache_size=200, coef0=0.0, degree=3, epsilon=0.00296, gamma='auto', kernel='rbf',
                  max_iter=-1, shrinking=True, tol=0.001, verbose=False),
          xgb.XGBRegressor(max_depth = 1, min_child_weight = 1, gamma = 0.02969, subsample = 0.9017,
                           colsample_bytree = 0.8, reg_alpha = 0.2902, reg_lambda = 0.1993, learning_rate = 0.21,
                           n_estimators = 22000, seed = 42, nthread = -1, silent = 1),
          linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=79)
          ]


rms = [rms_1_9, rms_1_7, rms_1_3, rms_1_3_dropped, rms_1_2, rms_12_31, rms_12_31_dropped, rms_AC]

#rms = [rms_1_9, rms_1_7, rms_AC]

# fit and predict using CV folds
# Strategy is:
# 1. Use kfold to split the training set S into \otimes_i S_i. Same with test set T.
# 2. Fit the models on S_{-j} = \otimes_{i \neq j} S_j
# 3. Use the fitted models as stage one to predict the response on S_j
# 4. Assemble the response on the whole of \otimes_j S_{-j} = S.
#   These are the training metafeatures in Breiman's CV stacking.
# 5. Use CV on the training set of metafeatures to tune the stage 2 regressor,
#   then use the whole set to fit.
# 6. Now use the stage one models to predict the response from the test data, eventually
#   reassembling predictions on the whole set T from those on the T_{-j}. These are the
#   test metafeatures.
# 7. Use the stage 2 regressor to predict the test response.

# define constants
n_data = 8
n_models = 8

n_train_obs = x_train[0].shape[0]
n_test_obs = x_test[0].shape[0]

n_feat = []
for i in range(0, n_data):
    n_feat.append(x_train[i].shape[1])

z_train = np.zeros((n_train_obs, n_data, n_models))
z_test = np.zeros((n_train_obs, n_data, n_models))

for i in range(0, n_data):
    print "Dataset: ", i
    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_train[i])):
        print "Fold: ", fold
        x_if, x_oof = x_train[i][train_idx], x_train[i][test_idx]
        # y_if, y_oof = y_train[i][train_idx], y_train[i][test_idx]
        y_if = y_train[train_idx]
        for a in range(0, n_models):
            print "Fitting model", a+1, "on fold ", fold+1, "of dataset ", i+1
            rms[i][a].fit(x_if,y_if)
            z_train[test_idx,i,a] = rms[i][a].predict(x_oof)

print("Metafeature set size:", z_train.shape)
print("Response set size:", y_train.shape)

z_train_mat = np.reshape(z_train, (n_train_obs, n_data * n_models))

print("Metafeature set size:", z_train_mat.shape)

# Fit the problem of predicting y_train from z_train

# Nonnegative least squares

ones_mat = np.ones(n_train_obs)

Z_mat = np.insert(z_train_mat, 0, ones_mat, axis=1)

nnLS = optimize.nnls(Z_mat, y_train)

#np.dot(Z_mat,nnLS[0])

rmse(y_train, np.dot(Z_mat,nnLS[0]))
# 0.10911506979852102

# nnLS[0].shape

# Ridge regression

def ridge_acc_test(_alpha):
# take argument alpha and return CV accuracy
    lr = linear_model.Ridge (alpha = _alpha)
    results = cross_val_score(lr, z_train_mat, y_train, cv=kfold)
    return (results.mean()*100.0, results.std()*100.0)

ridge_results_df = pd.DataFrame(dtype = 'float64')
for i in range(0, 20+1, 1):
    alpha = 0.0021 + 0.0001 * i
    ridge_results_df.loc[i, 'alpha'] = alpha
    (ridge_results_df.loc[i, 'accuracy'], ridge_results_df.loc[i, 'std dev']) = ridge_acc_test(alpha)

ridge_results_df.plot(x=['alpha'], y=['accuracy'])

ridge_results_df.sort_values(['accuracy'])
# 10  0.00910000000000000045  93.14983950432529979935  1.57673323515509156145
# 12  0.00929999999999999924  93.14984166317947256175  1.57765143359504400600
# 11  0.00919999999999999984  93.14984746912506352601  1.57719347288036915167

ridge_regr = linear_model.Ridge(alpha = 0.0025)
ridge_regr.fit(z_train_mat, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_ridge_pred = ridge_regr.predict(z_train_mat)
y_test = y_train
print("Ridge  score on training set: ", rmse(y_test, y_ridge_pred))
# ('Ridge  score on training set: ', 0.10991024414696889)

def lasso_acc_test(_alpha):
# take argument alpha and return CV accuracy
    lr = linear_model.Lasso(alpha = _alpha, max_iter=50000)
    results = cross_val_score(lr, z_train_mat, y_train, cv=kfold)
    return (results.mean()*100.0, results.std()*100.0)

lasso_results_df = pd.DataFrame(dtype = 'float64')
for i in range(0, 20+1, 1):
    alpha = 0.000001 + 0.000001 * i
    lasso_results_df.loc[i, 'alpha'] = alpha
    (lasso_results_df.loc[i, 'accuracy'], lasso_results_df.loc[i, 'std dev']) = lasso_acc_test(alpha)

lasso_results_df.plot(x = ['alpha'], y = ['accuracy'])

lasso_results_df.sort_values(['accuracy'])
#     alpha    accuracy   std dev
# 5   0.00086499999999999999  90.08332932890772326573  3.78972017105564784600
# 7   0.00086699999999999993  90.08336312239734411378  3.78893924519940217266
# 6   0.00086600000000000002  90.08353372924293012147  3.78886554867925706702

lasso_regr = linear_model.Lasso(alpha=0.000002, max_iter=50000)

lasso_regr.fit(z_train_mat, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_lasso_pred = lasso_regr.predict(z_train_mat)
y_test = y_train
print("Lasso score on training set: ", rmse(y_test, y_lasso_pred))
# ('Lasso score on training set: ', 0.1099079910656286)


def svr_acc_test(C_, eps):
# take arguments number of estimators and max depth and return CV accuracy
    svm_regr = svm.SVR(C=C_ , cache_size=200, coef0=0.0, degree=3, epsilon=eps, gamma='auto',
                        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    results = cross_val_score(svm_regr, z_train_mat, y_train, cv=kfold)
    return (results.mean()*100.0, results.std()*100.0)

svr_results_df = pd.DataFrame(dtype = 'float64')
for i in range(0, 20+1, 1):
    eps = 0.001 +  0.001*i
    svr_results_df.loc[i, 'epsilon'] = eps
    (svr_results_df.loc[i, 'accuracy'], svr_results_df.loc[i, 'std dev']) = svr_acc_test(1, eps)

svr_results_df.plot(x = ['epsilon'], y = ['accuracy'])

svr_results_df.sort_values(['accuracy'])
# 2   0.02999999999999999889  92.33626809813333125021  1.88572816154549727230
# 0   0.01000000000000000021  92.36310884668492349192  1.90759447853465347045
# 1   0.02000000000000000042  92.37055655885434646279  1.84256165766512536308

svr_results_df = pd.DataFrame(dtype = 'float64')
for i in range(0, 20+1, 1):
    C_ = 1 + 1*i
    svr_results_df.loc[i, 'C'] = C_
    (svr_results_df.loc[i, 'accuracy'], svr_results_df.loc[i, 'std dev']) = svr_acc_test(C_, 0.011)

svr_results_df.plot(x = ['C'], y = ['accuracy'])

svr_results_df.sort_values(['accuracy'])
# 11  31.0  92.81467797741636616138  1.79010998715076818932
# 9   29.0  92.81600618977522287878  1.79415049889271482897


lvl2_regr = svm.SVR(C=3, cache_size=200, coef0=0.0, degree=3, epsilon=0.011, gamma='auto',
                        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
lvl2_regr.fit(z_train_mat, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_svr_pred = lvl2_regr.predict(z_train_mat)
y_test = y_train
print("SVR score on MF training set: ", rmse(y_test, y_svr_pred))
# ('SVR score on MF training set: ', 0.11251539079223875)


# now make predictions over test set

z_test = np.zeros((n_test_obs, n_data, n_models))

for i in range(0, n_data):
    print "Dataset: ", i
    for a in range(0, n_models):
        print "Fitting model", a + 1, " on training dataset ", i + 1
        rms[i][a].fit(x_train[i], y_train)

        for fold, idx in enumerate(kfold.split(x_test[i])):
            print "Fold: ", fold
            train_idx, test_idx = idx
            x_oof = x_test[i][test_idx]
            print "Fitting model", a+1, "on fold ", fold+1, "of test dataset ", i+1
            z_test[test_idx,i,a] = rms[i][a].predict(x_oof)

z_test_mat = np.reshape(z_test, (n_test_obs, n_data * n_models))

### NNLS

ones_mat = np.ones(n_test_obs)
Z_test_mat = np.insert(z_test_mat, 0, ones_mat, axis=1)


y_stacking_pred = np.dot(Z_test_mat,nnLS[0])
y_stacking_pred = np.exp(y_stacking_pred)

pred_stacking_df = pd.DataFrame(y_stacking_pred, index=test_df["Id"], columns=["SalePrice"])
pred_stacking_df.to_csv('stacking_output-64.csv', header=True, index_label='Id')