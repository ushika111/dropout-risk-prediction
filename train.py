import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("online_learning_engagement_dropout_risk.csv")

# Encode target column
le = LabelEncoder()
df["dropout_risk"] = le.fit_transform(df["dropout_risk"])
# High=0, Low=1, Medium=2 (order may vary)

# Features + target
X = df.drop(["dropout_risk", "student_id"], axis=1)
y = df["dropout_risk"]

X = pd.get_dummies(X)  # handles course_category, device_type etc.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained successfully!")