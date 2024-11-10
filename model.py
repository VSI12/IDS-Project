import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn. neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


df = pd.read_csv('static/Files/NSL-KDD 2/train_data.csv', header=None, names=col_names)

df.head(5)

# Step 2: Preprocess the Data
# Separate features (X) and labels (y)
X = df.drop(columns=["label"])  # Replace "target" with your target column name
y = df["label"]

# Encode categorical columns in X if any
X = pd.get_dummies(X)
X_columns = X.columns

# Encode target labels (Normal/Anomaly)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


#models
modelRFC = RandomForestClassifier(n_estimators=100, random_state=42)
modelRFC.fit(X_train, y_train)

modelDTC = DecisionTreeClassifier(random_state=42)
modelDTC.fit(X_train, y_train)

modelKNN = KNeighborsClassifier()
modelKNN.fit(X_train, y_train)

modelGNB = GaussianNB ()
modelGNB.fit(X_train, y_train)