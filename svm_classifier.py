import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Define dataset paths
train_dir = "dataset/train"

# Image processing parameters
IMG_SIZE = 128  # Increase size for better feature extraction
categories = ["cats", "dogs"]

# Function to extract HOG features
def extract_hog_features(img):
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Function to load and preprocess data
def load_data(directory):
    X, y = [], []
    for label, category in enumerate(categories):  
        path = os.path.join(directory, category)
        for img_name in os.listdir(path):  
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
            features = extract_hog_features(img)  # Extract HOG features
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Load training data
X, y = load_data(train_dir)

# Normalize features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model with RBF kernel
svm_model = SVC(kernel="rbf", gamma="scale", C=10)  # C=10 improves margin
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=categories))

# Function to predict a single image
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = extract_hog_features(img)
    features = scaler.transform([features])  # Normalize
    prediction = svm_model.predict(features)
    return categories[prediction[0]]

# Example usage
sample_image = "dataset/test/cats/1.jpg"
print(f"Prediction: {predict_image(sample_image)}")

