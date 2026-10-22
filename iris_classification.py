
# =====================================================
# PROJECT 2: DATA CLASSIFICATION USING AI
# DecodeLabs Industrial Training Kit - Batch 2026
# =====================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# 1. LOAD THE DATASET
iris = load_iris()
X = iris.data
y = iris.target

print("=" * 50)
print("       IRIS FLOWER CLASSIFICATION USING AI")
print("=" * 50)

print("\nDataset loaded successfully!")
print("Total samples:", len(X))
print("Features:", iris.feature_names)
print("Flower classes:", list(iris.target_names))

# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. CREATE AND TRAIN THE KNN MODEL
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("\nKNN model trained successfully!")

# 5. PREDICT FLOWER CLASSES
y_pred = model.predict(X_test)

# 6. EVALUATE THE MODEL
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n" + "=" * 50)
print("             MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Weighted F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    zero_division=0
))

# 7. PREDICT A NEW FLOWER
# Measurements: sepal length, sepal width,
# petal length, petal width (all in cm)

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("=" * 50)
print("          NEW FLOWER PREDICTION")
print("=" * 50)

print("Measurements:", new_flower[0])
print("Predicted flower:", iris.target_names[prediction[0]])
print("\nProject completed successfully!")
