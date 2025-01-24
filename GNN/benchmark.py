import numpy as np
import time
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# Load data
print("Loading data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Data loaded successfully.")

# Split train data into train and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Data split completed.")

# Model 1: GradientBoostingClassifier
print("Initializing GradientBoostingClassifier...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_gb = MultiOutputClassifier(gb, n_jobs=-1)

# Training the GradientBoostingClassifier model
print("Training the GradientBoostingClassifier model...")
start_time = time.time()
model_gb.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# Evaluation on validation set for GradientBoostingClassifier
print("Evaluating GradientBoostingClassifier on validation set...")
val_predictions_gb = model_gb.predict(X_val)
val_recall_gb = recall_score(y_val, val_predictions_gb, average='micro')
val_precision_gb = precision_score(y_val, val_predictions_gb, average='micro')
val_f1_gb = f1_score(y_val, val_predictions_gb, average='micro')
val_accuracy_gb = accuracy_score(y_val, val_predictions_gb)
print(f"Validation Recall for GradientBoostingClassifier: {val_recall_gb:.4f}")
print(f"Validation Precision for GradientBoostingClassifier: {val_precision_gb:.4f}")
print(f"Validation F1 Score (Micro) for GradientBoostingClassifier: {val_f1_gb:.4f}")
print(f"Validation Accuracy for GradientBoostingClassifier: {val_accuracy_gb:.4f}")

# Model 2: Naive Bayes
print("Initializing Naive Bayes Classifier...")
nb = BernoulliNB()
model_nb = MultiOutputClassifier(nb, n_jobs=-1)

# Training the Naive Bayes model
print("Training the Naive Bayes model...")
start_time = time.time()
model_nb.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# Evaluation on validation set for Naive Bayes
print("Evaluating Naive Bayes on validation set...")
val_predictions_nb = model_nb.predict(X_val)
val_recall_nb = recall_score(y_val, val_predictions_nb, average='micro')
val_precision_nb = precision_score(y_val, val_predictions_nb, average='micro')
val_f1_nb = f1_score(y_val, val_predictions_nb, average='micro')
val_accuracy_nb = accuracy_score(y_val, val_predictions_nb)
print(f"Validation Recall for Naive Bayes: {val_recall_nb:.4f}")
print(f"Validation Precision for Naive Bayes: {val_precision_nb:.4f}")
print(f"Validation F1 Score (Micro) for Naive Bayes: {val_f1_nb:.4f}")
print(f"Validation Accuracy for Naive Bayes: {val_accuracy_nb:.4f}")

# Evaluation on test set for all models
print("Evaluating all models on test set...")
models = {
    "GradientBoostingClassifier": model_gb,
    "NaiveBayesClassifier": model_nb
}

for model_name, model in models.items():
    print(f"Evaluating {model_name} on test set...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    print(f"Prediction on test set for {model_name} completed in {time.time() - start_time:.2f} seconds.")

    # Calculate evaluation metrics
    print(f"Calculating evaluation metrics for {model_name}...")
    test_recall = recall_score(y_test, y_pred, average='micro')
    test_precision = precision_score(y_test, y_pred, average='micro')
    test_f1 = f1_score(y_test, y_pred, average='micro')
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Recall for {model_name}: {test_recall:.4f}")
    print(f"Test Precision for {model_name}: {test_precision:.4f}")
    print(f"Test F1 Score (Micro) for {model_name}: {test_f1:.4f}")
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")

print("Pipeline execution completed.")
