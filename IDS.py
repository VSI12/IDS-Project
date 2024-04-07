import pandas as pd
import numpy as np
import seaborn as sns
import imblearn
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time
import logging

from sklearn import preprocessing,svm,metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc
from sklearn.tree import DecisionTreeClassifier 
from sklearn. neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')


np.set_printoptions(precision=3)
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#load the dataset
train_url = 'NSL_KDD/NSL_KDD_Train.csv'

categorical_columns=['protocol_type', 'service', 'flag']
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def preprocess_dataset():
    df = pd.read_csv(train_url,header=None, names=col_names)
    print(df.shape)

    # Split the dataset into features (X) and target variable (y)
    X = df.drop('label', axis=1)
    y = df['label']
    print(X.shape)
    print(y.shape)


    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optionally, you can save the split datasets to CSV files
    X_train.to_csv('NSL-KDD 2/train_data.csv', index=False)
    X_test.to_csv('NSL-KDD 2/test_data.csv', index=False)
    y_train.to_csv('NSL-KDD 2/train_labels.csv', index=False)
    y_test.to_csv('NSL-KDD 2/test_labels.csv', index=False)

preprocess_dataset()