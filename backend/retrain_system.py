import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("--- INITIATING SYSTEM CALIBRATION INTERPOLATION ---")

if not os.path.exists("user_feedback_data.csv"):
    print("❌ ERROR: No feedback data collected yet. Scan faces and submit corrections first.")
    exit()

# 1. Concat both data files
train_df = pd.read_csv("train_features.csv")
feedback_df = pd.read_csv("user_feedback_data.csv")
combined_df = pd.concat([train_df, feedback_df], ignore_index=True)

print(f"Base instances: {len(train_df)} | User calibrated rows added: {len(feedback_df)}")

features = ['smile_w', 'smile_h', 'brow_h', 'eye_h', 'brow_i', 
            'mouth_drop', 'brow_v_left', 'brow_v_right', 'lip_dist', 'jaw_dist']

X = combined_df[features]
y = combined_df['label']

# 2. Fit transformation metrics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit strict Support Vector Boundaries
model = SVC(kernel='rbf', probability=True, C=12.0, class_weight='balanced')
model.fit(X_scaled, y)

# 4. Export artifacts
joblib.dump(model, "mood_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ SUCCESS: The system model weights have been fine-tuned to your hardware profile.")