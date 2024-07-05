import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew


def load_and_preprocess_data(filepath):
    print("Loading data...")
    train = pd.read_csv(filepath)
    print("Initial data shape:", train.shape)

    # Drop ID column in training dataset
    train = train.drop('Id', axis=1)

    # # Handling missing values for target variable 'SalePrice'
    # if train['SalePrice'].isna().any():
    #     train.dropna(subset=['SalePrice'], inplace=True)  # Dropping rows where 'SalePrice' is NaN
    #     print("Dropped rows where 'SalePrice' is NaN.")
    #
    # train['LogPrice'] = np.log(train['SalePrice'])
    # print("Now the histplot and proplot look much better!")
    # dist_price = sns.distplot(train['LogPrice'], fit=norm)
    # fig = plt.figure()

    # Drop outliers
    # train = train.drop(train[(train['1stFlrSF'] > 4000) & (train['SalePrice'] < 12.5)].index)

    # Handle categorical features with missing values
    categorical_fillna = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]
    for column in categorical_fillna:
        train[column] = train[column].fillna("None")

    # Replace missing values in some categorical features with the mode
    categorical_mode_fill = [
        'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 'SaleType'
    ]
    for column in categorical_mode_fill:
        train[column] = train[column].fillna(train[column].mode()[0])

    # Handle missing values in numerical features
    numerical_features = [
        'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
        'GarageCars', 'GarageArea'
    ]
    for feature in numerical_features:
        train[feature] = train[feature].fillna(train[feature].median())

    # Feature Engineering
    train['Total_Living_Area'] = train['1stFlrSF'] + train['2ndFlrSF']
    train['Total_Bathrooms'] = train['BsmtFullBath'] + train['BsmtHalfBath'] + train['FullBath'] + train['HalfBath']
    train['House_Age'] = train['YrSold'] - train['YearBuilt']
    train['Garage_Age'] = train['YrSold'] - train['GarageYrBlt']

    # data preprocessing
    # Select only the numeric features
    # numeric_features = train.select_dtypes(include=[np.number])

    # Calculate skewness for each numeric feature
    # skewness = numeric_features.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    #
    # # Identify skewed features (threshold can be adjusted as needed)
    # skewed_features = skewness[abs(skewness) > 0.75]
    # print(skewed_features)

    # Function to apply transformations
    # def transform_skewed_features(data, skewed_features):
    #     for feature in skewed_features.index:
    #         # Log transformation (add 1 to avoid log(0))
    #         if data[feature].min() >= 0:
    #             data[feature] = np.log1p(data[feature])
    #         else:
    #             data[feature] = np.log1p(data[feature] - data[feature].min() + 1)
    #     return data
    #
    # # Apply the transformation
    # data_transformed = transform_skewed_features(train.copy(), skewed_features)
    #
    # # Calculate skewness again for the transformed data
    # skewness_transformed = data_transformed[numeric_features.columns].apply(lambda x: skew(x.dropna())).sort_values(
    #     ascending=False)
    # print(skewness_transformed)

    # Scaling numerical features
    # scaler = StandardScaler()
    # numerical_vars = train.select_dtypes(include=['number']).columns.difference(['SalePrice'])
    # train[numerical_vars] = scaler.fit_transform(train[numerical_vars])
    #
    # # Encoding categorical variables
    # categorical_vars = train.select_dtypes(include=['object']).columns
    # train = pd.get_dummies(train, columns=categorical_vars, drop_first=True)

    # Identify numerical and categorical features
    numerical_vars = train.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    categorical_vars = train.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns

    numerical_vars_to_normalize = [col for col in numerical_vars if col != 'SalePrice']

    scaler = MinMaxScaler()
    # Fit the scaler on the numerical variables to normalize and transform the data
    train[numerical_vars_to_normalize] = scaler.fit_transform(train[numerical_vars_to_normalize])

    # encoding categorical features
    encoder = OneHotEncoder()
    X_categorical = encoder.fit_transform(train[categorical_vars]).toarray()
    X_categorical = pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out())

    # Combine numerical and encoded categorical features
    train = pd.concat([train[numerical_vars], X_categorical], axis=1)

    train = train.dropna()

    # Split features and target
    features = train.drop(['SalePrice'], axis=1)
    target = train['SalePrice']

    print("Data preprocessing complete.")
    print(train.shape)

    return features, target, features.columns
