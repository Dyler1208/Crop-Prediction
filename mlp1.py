import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

crops = pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\ML project 2\soil_measures.csv")
crops


# Features and target
X = crops[['N', 'P', 'K', 'ph']]
y = crops['crop']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model, scaler, and encoder saved successfully!")
