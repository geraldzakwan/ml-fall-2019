import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def get_excluded_features():
    df = pd.read_csv('CodeBook-SELECT.csv')
    excluded = []

    for i in range(0, 377):
        desc = df.iloc[i]['Description']
        varname = df.iloc[i]['VarName']

        if 'ISCO' in desc or 'ISCO' in varname:
            excluded.append(varname)

        elif 'ISIC' in desc or 'ISIC' in varname:
            excluded.append(varname)

        elif 'mth' in desc or 'mth' in varname:
            excluded.append(varname)

        elif 'coded' in desc or 'coded' in varname:
            excluded.append(varname)

    return excluded

def get_included_numeric_columns(df):
    # Find numeric columns
    num_col = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            num_col.append(col)

    print('NUM COL')
    print(len(num_col))

    incl_num_cols = []
    for col in num_col:
        if not df[col].isna().sum() > 10000:
            m = df[col].mean()
            if m >= 1.0:
                if not (m > 9000.0 and m < 10000.0):
                    incl_num_cols.append(col)

                    if m > 1990.0 and m < 2020.0:
                        df[col] = df[col] - 1990.0

    print('INCL COL')
    print(len(incl_num_cols))

    return df, incl_num_cols

def one_hot_encode(df, col_names):
    print(len(col_names))

    iter = 0
    new_cols = []
    for col in col_names:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            new_cols = new_cols + dummies.columns.tolist()
            df = pd.concat([df,dummies],axis=1)
            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)

            iter = iter + 1
            if (iter % 100 == 0):
                print(iter)

    return df, new_cols

def get_included_cat_cols(df, incl_num_cols):
    # TODO: Remove 999?

    cat_col = list(set(df.columns).difference(set(incl_num_cols)))

    print('CAT COL')
    print(len(cat_col))

    incl_cat_cols = []
    for col in cat_col:
        if len(df[col].unique()) < 11:
            incl_cat_cols.append(col)
    #     incl_cat_cols.append(col)

    print('INCL COL')
    print(len(incl_num_cols))

    print('There were {} columns before encoding categorical features'.format(df.shape[1]))
    df, incl_cat_cols = one_hot_encode(df, incl_cat_cols)
    print('There are {} columns after encoding categorical features'.format(df.shape[1]))

    return df, incl_cat_cols

def prepare_train(df, incl_num_cols, incl_cat_cols):
    # Drop 40% of the males to obtain balance
    sampling_percentage = 40
    dropped_indexes = df[df['gender_r_Male'] == 1].sample(frac=float(sampling_percentage/100)).index

    with open('dropped_indexes_' + str(sampling_percentage) + '.pickle', 'wb') as outfile:
        # dump information to that file
        pickle.dump(dropped_indexes, outfile)

    train_df = df.drop(dropped_indexes)
    print(len(train_df))
    print(len(train_df.columns))
    train_df = train_df[incl_num_cols + incl_cat_cols]

    df = df[incl_num_cols + incl_cat_cols]
    print(len(df))
    print(len(df.columns))

    return df, train_df

def train_and_eval(df, train_df):
    X_train = train_df.drop(['job_performance'], axis=1).values
    y_train = train_df['job_performance'].values

    # from sklearn.model_selection import GridSearchCV
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10, 20],
    #     'max_features': ['auto'],
    #     'min_samples_leaf': [50, 100],
    #     'min_samples_split': [100, 200],
    #     'n_estimators': [20, 50]
    # }
    # # Create a based model
    # rf = RandomForestRegressor(random_state = 40)
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    # grid_search.fit(X_train, y_train)

    # print(grid_search.best_params_)
    # clf = grid_search.best_estimator_

    # Train the model using the training sets
    clf = RandomForestRegressor(max_depth = 10, random_state = 28)
    print(clf.get_params())
    clf.fit(X_train, y_train)

    X_train = df.drop(['job_performance'], axis=1).values
    y_train = df['job_performance'].values

    y_pred = clf.predict(X_train)

    print('Mean: ' + str(np.mean(y_pred)))
    print('Variance: ' + str(np.var(y_pred)))

    return clf, mean_squared_error(y_train, y_pred)

def main_1(model_name):
    df = pd.read_csv('hw4-trainingset-gd2551.csv')
    df = df.drop(['uni', 'row'] + get_excluded_features(), axis=1)

    # Impute columns simply with mode
    df = df.fillna(df.mode().iloc[0])

    df, incl_num_cols = get_included_numeric_columns(df)

    df, incl_cat_cols = get_included_cat_cols(df, incl_num_cols)

    df, train_df = prepare_train(df, incl_num_cols, incl_cat_cols)

    clf, mse = train_and_eval(df, train_df)

    print('MSE: ' + str(mse))

    with open(model_name, 'wb') as outfile:
        pickle.dump(clf, outfile)

    return df.columns

def main_2(model_name, train_cols):
    df = pd.read_csv('hw4-testset-gd2551.csv')
    df = df.drop(['uni', 'row'] + get_excluded_features(), axis=1)

    # Impute columns simply with mode
    df = df.fillna(df.mode().iloc[0])

    df, incl_num_cols = get_included_numeric_columns(df)

    df, incl_cat_cols = get_included_cat_cols(df, incl_num_cols)

    for missing_col in list(set(train_cols).difference(set(df.columns))):
        df[missing_col] = np.zeros(24500, dtype='int')

    df = df[train_cols]

    X_test = df.drop(['job_performance'], axis=1).values

    with open(model_name, 'rb') as infile:
        clf = pickle.load(infile)

    y_pred = clf.predict(X_test)

    print('Mean: ' + str(np.mean(y_pred)))
    print('Variance: ' + str(np.var(y_pred)))

    return df.columns

if __name__ == '__main__':
    x = main_1('test_model.pickle')
    a = main_2('test_model.pickle', x)

    # print('NUMS')
    print(set(x).difference(set(a)))
    # print('----')
    # print(set(a).difference(set(x)))
    # print('----')
    #
    # print('CATS')
    # print(set(y).difference(set(b)))
    # print('----')
    # print(set(b).difference(set(y)))
    # print('----')
