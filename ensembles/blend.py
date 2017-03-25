# Python script for kaggle house price challenge

# import functions
import numpy as np
import pandas as pd
pd.set_option('display.precision',20)
from scipy import stats

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
# train_df = pd.read_csv("../train-ppd.csv")
# test_df = pd.read_csv("../test-ppd.csv")

train_df = pd.read_csv("train-1-9-delout.csv")
test_df = pd.read_csv("test-1-9-delout.csv")

# train_df = pd.read_csv("AC_train.csv")
# test_df = pd.read_csv("AC_test.csv")
# label_df = pd.read_csv("AC_label.csv")

# Set up predictors and response

y_train = train_df['LogSalePrice'].values
x_train = train_df.drop(['Id','LogSalePrice'],axis=1).values
x_test = test_df.drop(['Id'],axis=1).values

# y_train = label_df['SalePrice'].values
# x_train = train_df.drop(['Id'],axis=1).values
# x_test = test_df.drop(['Id'],axis=1).values


print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

#  Ridge regression

ridge_regr = linear_model.Ridge(alpha = 55)

ridge_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_ridge_pred = ridge_regr.predict(x_train)
y_test = y_train
print("Ridge score on training set: ", rmse(y_test, y_ridge_pred))
# ('Ridge score on training set: ', 0.096912731484273193)

#  Lasso regression

lasso_regr = linear_model.Lasso(alpha=0.0006, max_iter=50000)

lasso_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_lasso_pred = lasso_regr.predict(x_train)
y_test = y_train
print("Lasso score on training set: ", rmse(y_test, y_lasso_pred))
# ('Lasso score on training set: ', 0.10122553836843487)

# Elastic net

elnet_regr = linear_model.ElasticNet(alpha = 0.0011, l1_ratio=0.5, max_iter=15000, random_state=7)

elnet_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_elnet_pred = elnet_regr.predict(x_train)
y_test = y_train
print("Elastic Net score on training set: ", rmse(y_test, y_elnet_pred))
# ('Elastic Net score on training set: ', 0.10126617210593648)

# Random Forest

rf_regr = RandomForestRegressor(n_estimators = 700, max_depth = 25, random_state = 7)

rf_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_rf_pred = rf_regr.predict(x_train)
y_test = y_train
print("RF score on training set: ", rmse(y_test, y_rf_pred))
# ('RF score on training set: ', 0.050259305243602503)

# SVR

svm_regr = svm.SVR(C=5, cache_size=200, coef0=0.0, degree=3, epsilon=0.034, gamma=0.0004,
                        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

svm_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_svm_pred = svm_regr.predict(x_train)
y_test = y_train
print("SVR score on training set: ", rmse(y_test, y_svm_pred))
# ('SVR score on training set: ', 0.087579619440749337)

# XGB

xgb_regr = xgb.XGBRegressor(
    max_depth = 1,
    min_child_weight = 0.5,
    gamma = 0,
    subsample = 1,
    colsample_bytree = 0.6,
    reg_alpha = 0.45,
    reg_lambda = 0.2,
    learning_rate = 0.05,
    n_estimators = 6100,
    seed = 42,
    nthread = -1,
    silent = 1)

xgb_regr.fit(x_train, y_train)

# Run prediction on training set to get a rough idea of how well it does.
y_xgb_pred = xgb_regr.predict(x_train)
y_test = y_train
print("XGB score on training set: ", rmse(y_test, y_xgb_pred))
# ('XGB score on training set: ', 0.023789953069865782)


# assemble test set predictions
y_ridge_pred = ridge_regr.predict(x_test)
y_lasso_pred = lasso_regr.predict(x_test)
y_elnet_pred = elnet_regr.predict(x_test)
y_rf_pred = rf_regr.predict(x_test)
y_svr_pred = svm_regr.predict(x_test)
y_xgb_pred = xgb_regr.predict(x_test)

norm = ( 1 / 0.097 + 1 / 0.10 + 1 / 0.10  + 1 / 0.05 + 1 / 0.087 + 1 / 0.024 )

# y_pred_blend = ( y_ridge_pred / 0.097 + y_lasso_pred / 0.10 + y_elnet_pred / 0.10
#                  + y_rf_pred / 0.05 + y_svr_pred / 0.087 + y_xgb_pred / 0.024) / norm

y_pred_blend = (y_lasso_pred + y_xgb_pred) / 2

y_pred_blend = np.exp(y_pred_blend)

pred_blend_df = pd.DataFrame(y_pred_blend, index=test_df["Id"], columns=["SalePrice"])
pred_blend_df.to_csv('blend_output-2.csv', header=True, index_label='Id')