import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset (THIS is your correct file)
df = pd.read_csv("online_learning_engagement_dropout_risk.csv")

# Check columns (important for debugging)
print(df.columns)

# CHANGE THIS if needed after you see columns
X = df.drop("dropout_risk", axis=1)
y = df["dropout_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")