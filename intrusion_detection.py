import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier 
import pickle




train_url = 'NSL_KDD/NSL_KDD_Train.csv'
test_url = 'NSL_KDD/NSL_KDD_Test.csv'


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


#load nsl dataset and preprocess it
def load_dataset():

    df = pd.read_csv(train_url, header=None, names=col_names)
    df_test = pd.read_csv(test_url, header=None, names = col_names)

    print('Dimensions of the Training set:', df.shape)
    print('Dimensions of the test set: ', df_test.shape)

    df.head()

    print(df.head(10))

    print('Label distribution Training set:')
    print(df['label'].value_counts())
    print()
    print('Label distribution Test set:')
    print(df_test['label'].value_counts())

    

    print('Training set:')
    for col_name in df.columns:
     if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    print()
    print('Distribution of categories in service:')
    print(df['service'].value_counts().sort_values(ascending=False).head())

    print('Test set:')
    for col_name in df_test.columns:
        if df_test[col_name].dtypes == 'object' :
            unique_cat = len(df_test[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


    categorical_columns=['protocol_type', 'service', 'flag']

    df_categorical_values = df[categorical_columns]
    testdf_categorical_values = df_test[categorical_columns]

    df_categorical_values.head()
    print(df_categorical_values.head())
    
    for feature in categorical_columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

        print('df[feature]')

        print(df[feature])


    return df


def train_model():
    df = load_dataset()

    X = df.drop(columns=['label'])
    y = df['label']
    print('x and y')
    print(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Save trained model
    with open('intrusion_detection_model.pkl', 'wb') as file:
        pickle.dump(clf, file)


print(train_model())