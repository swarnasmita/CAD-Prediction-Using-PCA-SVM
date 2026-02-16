
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib

# Load the dataset
data = pd.read_csv('heart.csv')  # Update this path with your file's location

# Convert the data to a numerical array (excluding non-numeric columns)
data_array = data.select_dtypes(include=[np.number]).values

# Standardize the data (zero mean, unit variance)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_array)

# Create a PCA object and fit it to the standardized data
pca = PCA()
pca.fit(standardized_data)

# Explained variance (percentage of variance explained by each component)
explained_variance = pca.explained_variance_ratio_ * 100

# Cumulative variance explained
cumulative_variance = np.cumsum(explained_variance)

# Find the number of components to reach a threshold of cumulative variance (e.g., 75%)
threshold = 75
num_components = np.argmax(cumulative_variance >= threshold) + 1
print(f"Number of components to reach {threshold}% variance: {num_components}")

# Select the top 2 principal components for visualization
k = 2
principal_components = pca.components_[:k]

# Project the data onto the new subspace (reduced dataset using only 2 components)
projected_data = standardized_data.dot(principal_components.T)

# Assuming the last column is the label (target variable)
labels = data['target'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(projected_data, labels, test_size=0.3, random_state=42)

# Stratified K-Fold Cross-Validation to handle imbalanced classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create an SVM classifier with a linear kernel
svm = SVC(kernel='rbf', class_weight='balanced')

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(svm, param_grid, cv=cv)
grid_search.fit(X_train, y_train)
print(f"Best parameters from GridSearch: {grid_search.best_params_}")

# Train the SVM classifier with the best parameters
svm_model = grid_search.best_estimator_

# Train the SVM classifier
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM classifier on reduced data: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report for SVM Classifier:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix for SVM Classifier')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, svm_model.decision_function(X_test))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, svm_model.decision_function(X_test))
average_precision = average_precision_score(y_test, svm_model.decision_function(X_test))

plt.figure()
plt.plot(recall, precision, color='b', label=f'Average Precision = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# Save the trained SVM model
joblib.dump(svm_model, 'svm_model.pkl')
print("SVM model saved as svm_model.pkl")

# --- Plotting Decision Boundary ---
# Create a meshgrid for the 2D space
h = .02  # Step size in the mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap='coolwarm')
plt.colorbar()  # Color scale for predicted labels

# Plot the training and test points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolor='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', s=100, label='Test Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with Contour Plot')
plt.legend()
plt.grid(True)
plt.show()
