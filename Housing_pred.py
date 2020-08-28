import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import datetime

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm

from keras.layers import Dense, Activation
from keras.models import Sequential

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

#Sources
# https://www.kaggle.com/rislam173/house-prices-4-feature-scaling
# https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

# Load the files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Get number of observations for test and train
print([len(x) for x in [train_df, test_df]])

# Combine it into one large file for data exploration and cleaning
combined_df = pd.concat([train_df, test_df])

# Get a first view
print(combined_df)

# Classify int variables into category if needed
combined_df["MSSubClass"] = combined_df["MSSubClass"].astype("category")
combined_df["MoSold"] = combined_df["MoSold"].astype("category")

# Quick look at potential missing values
print(combined_df.info())

# List of cols with missing values
print([col for col in combined_df.columns if combined_df[col].isnull().any()])

# Handling missing values correctly might greatly help us, we spend some time here
missing  = ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
            'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
            'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
            'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
            'GarageArea', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SalePrice']

# List of cols with missing values in test set only
missing_test = [col for col in test_df.columns if test_df[col].isnull().any()]

print([x for x in missing if x not in missing_test]) # Only "Electrical" is only missing in train set, so we do nothing

# Categorical data impute with mode of neighborhood and MSSubClass or just mode of own column if missing
missing_vals = ["MSZoning", "Alley", "Utilities", 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2',
                "Electrical",'KitchenQual', 'Functional','GarageType',"SaleType", 'GarageFinish','GarageQual','GarageCond',
                'Exterior1st', 'Exterior2nd','FireplaceQu', "PoolQC", "Fence", "MiscFeature"]

for missing_val in missing_vals:
    try:
        combined_df[missing_val] = combined_df.groupby(['MSSubClass', "Neighborhood"])[missing_val].transform(lambda x: x.fillna(x.mode()[0]))
    except:
        combined_df[missing_val].fillna((combined_df[missing_val].mode()[0]), inplace=True)

# Add "Other" category as few elements are missing
combined_df["PoolQC"] = combined_df["PoolQC"].fillna("Other")

# Continuous data
missing_vals = ["LotFrontage", 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF1','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                'GarageCars', 'GarageArea',]
impute_vals = ["LotConfig" ,"Neighborhood",'BsmtFinType1', 'BsmtFinType2','BsmtQual', 'BsmtQual', 'BsmtQual',
               'GarageType', 'GarageType']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df.groupby(impute_val)[missing_val].transform('mean'))

# Continuous impute data based on other continuous data
missing_vals = ['GarageYrBlt']
impute_vals = ['YearBuilt']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df[impute_val])

# Fill all leftovers with mean
for missing_val in combined_df.columns.values.tolist():

    if missing_val == "SalePrice":
        pass

    else:
        try:
            combined_df[missing_val] = combined_df[missing_val].fillna(combined_df[missing_val].mean())
        except:
            pass

# List of cols with missing values
print([col for col in combined_df.columns if combined_df[col].isnull().any()])

# Quick look at potential missing values
print(combined_df.info())

# Get a sense of the data
print(combined_df.describe())

# Add and change some variables, namely the "Year" ones as it would be better to have them as "Age"
year = datetime.date.today().year
combined_df["AgeSold"] = int(year) - combined_df["YrSold"].astype(int)
combined_df["AgeGarage"] = int(year) - combined_df["GarageYrBlt"].astype(int)
combined_df["AgeBuilt"] = int(year) - combined_df["YearBuilt"].astype(int)

# Add some features related to total area of the house
combined_df['TotalArea'] = combined_df['TotalBsmtSF'] + combined_df['1stFlrSF'] + combined_df['2ndFlrSF'] + combined_df['GrLivArea'] +combined_df['GarageArea']
combined_df['Bathrooms'] = combined_df['FullBath'] + combined_df['HalfBath']*0.5
combined_df['Year average']= (combined_df['YearRemodAdd']+combined_df['YearBuilt'])/2




"""

# Check the sale price distribution by different types of variables
for element in ["MSSubClass", "MSZoning", "Utilities", "HouseStyle", "Neighborhood", "PoolQC", "SaleType"]:
    cat_plot = sns.catplot(y="SalePrice", x= element, kind="swarm", legend="full", data=combined_df, height=4.5, aspect=3 / 3,);
    cat_plot.set_xticklabels(rotation=90)

for element in ["1stFlrSF", "LotArea", "OverallQual", "OverallCond", "YearBuilt","ExterQual", "YrSold"]:
    re_plot = sns.relplot(y="SalePrice", x= element, legend="full", data=combined_df, height=4.5, aspect=3 / 3,);
    re_plot.set_xticklabels(rotation=90)


# Correlation matrix
corr_mat = combined_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_mat, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

"""




# Get dummies for our data set
combined_df = pd.get_dummies(combined_df)

# Split the data set so to build our model
train_df = combined_df[combined_df["SalePrice"] > 0 ]
test_df = combined_df[combined_df["SalePrice"].isna() ]
test_df = test_df.drop(["SalePrice"], axis = 1)

# Create the X and y sets
X_train_df = train_df.drop(["SalePrice"], axis = 1)
y_train_df = train_df[["Id" ,"SalePrice"]]

# Log transform the SalePrice as it is skewed
y_train_df["SalePrice"] = np.log1p(y_train_df["SalePrice"])

# Set the ID col as index
for element in [X_train_df, y_train_df, test_df]:
    element.set_index('Id', inplace = True)




# Scale the data and use RobustScaler to minimize the effect of outliers
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
scaler = RobustScaler()

# Scale the X_train set
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_train_df = pd.DataFrame(X_train_scaled, index = X_train_df.index, columns= X_train_df.columns)

# Scale the X_test set
X_test_scaled = scaler.fit_transform(test_df.values)
X_test_df = pd.DataFrame(X_test_scaled, index = test_df.index, columns= test_df.columns)



"""
# Feature selection (only keep variables with some variance)
threshold_n=0.55
sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
sel_var=sel.fit_transform(X_train_df)

# Create the new datasets
X_train_df = X_train_df[X_train_df.columns[sel.get_support(indices=True)]]
X_test_df = X_test_df[X_test_df.columns[sel.get_support(indices= True)]]

# Check what we have
print(X_train_df.info())
"""

# Modeling

# Split our training set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.05, random_state=23)

# 0. REGULARIZATION WITH ELASTIC NET

alphas = [0.000542555]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]

elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

# Fit the model to the data
estc_reg = elastic_cv.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = estc_reg.predict(X_test)
print("ElasticRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(estc_reg.predict(X_test_df))
my_pred_estc = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_estc.to_csv('pred_estc.csv', index=False)


# 0. REGULARIZATION WITH LASSO

parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}

# Instantiate reg for gridsearch
lasso=Lasso()
lasso_reg = GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

# Instantiate new lasso reg with best params
lasso_reg = Lasso(alpha= 0.0009)

# Fit the model to the data
lasso_reg.fit(X_train,y_train)

# Predict on the test set from our training set
y_pred = lasso_reg.predict(X_test)
print("LassoRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(lasso_reg.predict(X_test_df))
my_pred_lasso = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_lasso.to_csv('pred_lasso.csv', index=False)



# 1. RANDOM FOREST

"""
# Grid search for best params

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Create a based model
rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)
"""

# Create a random forest with best parameters
rf_reg = RandomForestRegressor(bootstrap =  True, max_depth = 80, max_features = 'auto', min_samples_leaf = 3,
                               min_samples_split = 8, n_estimators = 300, n_jobs=-1, random_state=12)

# Fit the model to the data
rf_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred_rf = rf_reg.predict(X_test)
print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_rf)))

# Create predictions
predictions = np.exp(rf_reg.predict(X_test_df))
my_pred_rf = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_rf.to_csv('pred_rf.csv', index=False)


# 2. ADA BOOST

# Grid search for best params
param_grid = {
 'n_estimators': [50, 100, 200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'loss' : ['linear', 'square', 'exponential']
 }

# Create a based model
ab_reg = AdaBoostRegressor()

"""
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = ab_reg, param_grid = param_grid,
                          cv = 4, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)
"""

# Create a random forest with best parameters
ab_reg = AdaBoostRegressor(learning_rate =1, loss = 'exponential', n_estimators =  50, random_state= 12)

# Fit the model to the data
ab_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred_ab = ab_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ab)))

# Create predictions
predictions = np.exp(ab_reg.predict(X_test_df))
my_pred_ab = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_ab.to_csv('pred_ab.csv', index=False)



# 3. XGBOOST

"""
# Grid search for best params
param_grid = {'max_depth':[3,4],
          'learning_rate':[0.01,0.03],
          'min_child_weight':[1,3],
          'reg_lambda':[0.1,0.5],
          'reg_alpha':[1,1.5],      
          'gamma':[0.1,0.5],
          'subsample':[0.4,0.5],
         'colsample_bytree':[0.4,0.5],
}

# Create a based model
reg = XGBRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = reg, param_grid = param_grid,
                          cv = 4, n_jobs = -1, verbose = True)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)
"""

# Create a regressor with best parameters
"""
reg = XGBRegressor(colsample_bytree = 0.7, learning_rate= 0.05, max_depth = 5,
 min_child_weight = 4, n_estimators = 500, nthread = 4, objective = 'reg:linear', silent = 1, subsample= 0.7, random_state= 10)
"""

xgb_reg = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

# Fit the model to the data
xgb_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = xgb_reg.predict(X_test)
print("XGBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(xgb_reg.predict(X_test_df))
my_pred_xgb = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_xgb.to_csv('pred_xgb.csv', index=False)


# 5. NEURAL NETWORK

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 320))

# Adding the second hidden layer
model.add(Dense(units = 320, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 300, activation = 'relu'))

# Adding the fourth hidden layer
model.add(Dense(units = 600, activation = 'relu'))

# Adding the fifth hidden layer
model.add(Dense(units = 800, activation = 'relu'))

# Adding the sixth hidden layer
model.add(Dense(units = 300, activation = 'relu'))

# Adding the seventh hidden layer
model.add(Dense(units = 300, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 200)

y_pred = model.predict(X_test)
print("ANNRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(model.predict(X_test_df))
predictions = np.concatenate( predictions, axis=0 )
my_pred_ann = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_ann.to_csv('pred_ann.csv', index=False)



# 6. LIGHTGBM

lgbm_reg = lgbm.LGBMRegressor(
    objective='regression',
    num_leaves=4,
    learning_rate=0.01,
    n_estimators=5000,
    max_bin=200,
    bagging_fraction=0.75,
    bagging_freq=5,
    bagging_seed=7,
    feature_fraction=0.2,
    feature_fraction_seed=7,
    verbose=-1,
    #min_data_in_leaf=2,
    #min_sum_hessian_in_leaf=11
)

# Fit the model to the data
lgbm_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = lgbm_reg.predict(X_test)
print("LGBMRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(lgbm_reg.predict(X_test_df))
my_pred_lgbm = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_lgbm.to_csv('pred_lgbm.csv', index=False)

# 8. SVM

svr_reg = make_pipeline(RobustScaler(), SVR(
    C=20,
    epsilon=0.008,
    gamma=0.0003,
))

# Fit the model to the data
svr_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = svr_reg.predict(X_test)
print("SVRRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(svr_reg.predict(X_test_df))
my_pred_svr = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_svr.to_csv('pred_svr.csv', index=False)


# 9. STACKED REGRESSION

# Stack the best models for best answer
stregr = StackingRegressor(regressors=[xgb_reg, estc_reg, lasso_reg, lgbm_reg],
                           meta_regressor=lgbm_reg, use_features_in_secondary=True
                          )
# Fit the model
stack_reg=stregr.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = stack_reg.predict(X_test)
print("StackedRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(stack_reg.predict(X_test_df))
my_pred_stacked = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_stacked.to_csv('pred_stacked.csv', index=False)



# Visualize plots of sales prices

x = my_pred_xgb["SalePrice"]
train_df = pd.read_csv("train.csv")
y = train_df["SalePrice"]

# Method 1: on the same Axis
#sns.distplot(x, color="skyblue", label="Predicitions")
sns.distplot(y, color="red", label="Training")
plt.legend()

plt.show()
