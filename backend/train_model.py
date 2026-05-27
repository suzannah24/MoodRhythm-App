import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the 10-feature data
train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

# 2. Define all 10 features
features = ['smile_w', 'smile_h', 'brow_h', 'eye_h', 'brow_i', 
            'mouth_drop', 'brow_v_left', 'brow_v_right', 'lip_dist', 'jaw_dist']

X_train = train_df[features]
y_train = train_df['label']
X_test = test_df[features]
y_test = test_df['label']

# 3. Scale the data (Mandatory for 10+ features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train with 'Balanced' weights to force the AI to see 'Angry' and 'Sad'
model = SVC(kernel='rbf', probability=True, C=10.0, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 5. Report
y_pred = model.predict(X_test_scaled)
print(f"FYP 2 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# 6. Save BOTH files
joblib.dump(model, "mood_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print("SUCCESS: 10-Feature Model and Scaler Saved.")