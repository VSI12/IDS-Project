import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('../dataset/KDDTrain+.txt', header=None)

# Define feature columns and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
with open('model/ids_model.pkl', 'wb') as f:
    pickle.dump(model, f)
