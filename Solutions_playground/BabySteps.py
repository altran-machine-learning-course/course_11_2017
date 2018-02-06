import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys


#Ignore Warnings - save some confusion
import warnings
warnings.filterwarnings('ignore')

#Pandas more columns
pd.options.display.max_columns = None
pd.set_option('display.max_columns', None)

# Add input as import path
sys.path.insert(0,'../input')

# Plot style
plt.style.use('fivethirtyeight')

# Import the data from the dataset
train_data = pd.read_csv('../input/train.csv',index_col='id')
test_data = pd.read_csv('../input/test.csv',index_col='id')
train_data_orig = train_data.copy()
X_train_orig = train_data_orig.loc[:,train_data_orig.columns!="survived"]
Y_train_orig = train_data_orig.loc[:,"survived"]
X_test_orig = test_data.loc[:, test_data.columns!="surived"]

def simplify_fares(df):
    df.fare = df.fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.fare, bins, labels=group_names)
    df.fare = categories
    return df

def simplify_cabins(df):
    df["nr_cabins"] = df.cabin.apply(lambda x : len(str(x).split()))
    df.cabin = df.cabin.fillna('N')
    df["cabin_number"] = df.cabin.apply(lambda x: str(x).split()[0][1:])
    df["cabin_number"] = df["cabin_number"].apply(lambda x: x if (len(str(x)) > 0) else np.nan)
    #df["cabin_number"].replace(r'\s+', np.nan, regex=True)
    df.cabin = df.cabin.apply(lambda x: str(x)[0])
    return df


def format_name(df):
    df['lname'] = df.name.apply(lambda x: x.split(' ')[0])
    df['lname'].fillna(' ')
    df['nameprefix'] = df.name.apply(lambda x: x.split(' ')[1])
    df['nameprefix'].fillna(' ')
    return df


def drop_features(df):
    return df.drop(['ticket', 'name', 'embarked', 'home.dest'], axis=1)

def transform_features(df):
    #df = simplify_fares(df) --> Don't want to simplify fares
    df = simplify_cabins(df)
    df = format_name(df)
    df = drop_features(df)
    return df


train_data = transform_features(train_data)
test_data  = transform_features(test_data)








from sklearn import preprocessing
def encode_features(df_train, df_test):
    # Removed 'fares'
    features = [ 'cabin', 'sex', 'lname', 'nameprefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

train_data, test_data = encode_features(train_data, test_data)




def fill_missing_data(df_train,df_test):
    # Added fare to inputer
    features = ['age', 'fare', 'cabin_number']
    df_combined = pd.concat([df_train[features], df_test[features]])
    df_imputer = preprocessing.Imputer()
    df_imputer.fit(df_combined[features])
    df_train[features] = df_imputer.transform(df_train[features])
    df_test[features] = df_imputer.transform(df_test[features])
    return df_train, df_test

train_data_missing, test_data_missing = train_data.copy(), test_data.copy()

train_data_unimputed, test_data_unimputed = \
    train_data_missing.dropna(axis=0, how='any'), test_data_missing.dropna(axis=0, how='any')
train_data,test_data = fill_missing_data(train_data,test_data)

def get_X_Y_pair(df):
    features = df.columns.values
    x_features = [f for f in features if f!='survived']
    return df[x_features], df['survived']

def scale_data(df_train, df_test):
    df_combine = pd.concat([df_train, df_test])
    features = df_train.columns.values
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_combine)
    return scaler.transform(df_train), scaler.transform(df_test)

x_train, y_train = get_X_Y_pair(train_data)
x_test, y_test = get_X_Y_pair(test_data)

x_train_minimal, y_train_minimal = get_X_Y_pair(train_data_unimputed)
x_test_minimal, y_test_minimal = get_X_Y_pair(test_data_unimputed)

#not pandas after this
# print(type(x_train), type(x_test), type(x_train_minimal), type(x_test_minimal))

x_train_minimal, x_test_minimal = scale_data(x_train_minimal, x_test)
x_train_unscaled,x_test_unscaled = x_train.copy(), x_test.copy()
x_train, x_test = scale_data(x_train,x_test)

import joblib
joblib.dump((x_train,y_train),"traindata.pkl")
joblib.dump((x_test, y_test), "testdata.pkl")
joblib.dump((x_train_unscaled, y_train), "traindata_unscaled.pkl")
joblib.dump((x_test_unscaled, y_test), "testdata_unscaled.pkl")
joblib.dump((x_train_minimal, y_train_minimal), "traindata_minimal.pkl")
joblib.dump((x_test_minimal, y_test), "testdata_minimal.pkl")


print(x_train.shape, y_train.shape)
print(x_train_minimal.shape, y_train_minimal.shape)
##Perceptron
# import sys
# sys.path.insert(0,'../input')
#
# from utils import accuracy_score_numpy
# def train_test_accuracy(clf,x_train, y_train,x_test):
#     clf.fit(x_train, y_train)
#     y_test = clf.predict(x_test)
#     perc = accuracy_score_numpy(y_test)
#     return perc

# from sklearn.linear_model import Perceptron
# percep = Perceptron()
# acc = train_test_accuracy(percep, x_train_minimal, y_train_minimal,x_test)
# print("Accuracy for perceptron is  ", acc)


# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=10000, n_jobs=4, criterion="entropy")
# acc = train_test_accuracy(rf, x_train_unscaled, y_train, x_test_unscaled)
# print("Accuracy for RF is  ", acc)

# from tpot import TPOTClassifier
# tpot = TPOTClassifier(generations=20, population_size=50, verbosity=2)
# acc = train_test_accuracy(tpot, x_train_unscaled, y_train, x_test_unscaled)
# print("Accuracy for TPOT is  ", acc)


# import autosklearn.classification
# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600,per_run_time_limit=180)
# acc = train_test_accuracy(automl, x_train_unscaled, y_train, x_test_unscaled)
# print("Accuracy for RF is  ", acc)

#import _pickle as pkl
#pkl.dumps((x_test, y_test), open("testdata.pkl","w"))
#pkl.dumps((x_train, y_train), open("traindata.pkl","w"))
