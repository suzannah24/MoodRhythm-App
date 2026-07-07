import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# 1. Load your live dataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('user_feedback_data.csv')

# Drop low-sample classes to prevent splitting errors (surprise, disgust)
df = df[df['label'].isin(['happy', 'sad', 'angry', 'neutral'])]

X = df.drop(columns=['label'])
y = df['label']

# Encode categorical text labels to numbers for MAE calculation
label_mapping = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3}
y_encoded = y.map(label_mapping)

# 2. Split into Train and Test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# 3. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize the 4 models
models = {
    "Support Vector Machine (SVM)": SVC(kernel='rbf', class_weight='balanced', probability=True),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes Classifier": GaussianNB()
}

# 5. Evaluate and Print Results
print(f"{'Model Algorithm Evaluation Type':<35} | {'Accuracy':<8} | {'F1-Score':<8} | {'MAE':<8}")
print("-" * 70)

for name, model in models.items():
    # Use scaled features for distance/margin based models, raw for Naive Bayes
    if name == "Naive Bayes Classifier":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"{name:<35} | {acc:.4f}   | {f1:.4f}   | {mae:.4f}")