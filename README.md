Project Overview: 
This project builds a clinically inspired machine learning pipeline for early Coronary Artery Disease detection, leveraging Principal Component Analysis (PCA) to distill high-dimensional health data into meaningful components and an optimized Support Vector Machine (SVM) to perform reliable risk classification. The framework balances interpretability, computational efficiency, and predictive strength.

The system achieved:
  1. Accuracy: 92.21%
  2. ROC-AUC: 0.95

Dataset Information:
  1. Total Samples: 1025 patients
  2. Features: 14 clinical attributes
  3. Target: Presence or absence of heart disease
  4. Source: Cleveland Heart Disease Dataset (Kaggle)

Key attributes include:
  1. Age 
  2. Sex 
  3. Chest pain type (cp) 
  4. Resting blood pressure (trestbps) 
  5. Serum cholesterol in mg/dl (chol) 
  6. Fasting blood sugar (fbs) 
  7. Resting electrocardiographic result (restecg) 
  8. Maximum heart rate achieved (thalach) 
  9. Exercise induced angina (exang) 
  10. ST depression induced by exercise relative to rest (oldpeak) 
  11. Slope of the peak exercise ST segment (slope) 
  12. Number of major vessels colored by fluoroscopy (ca) 
  13. Inherited blood disorder, Thallasemia (thal) 
  14. Buildup of plaque in the walls of blood vessels (target)

Methodology:
  1. Data Preprocessing: 
     - Handling inconsistencies
     - Z-score standardization
     - Train-test split (70:30)
       
  2. Dimensionality Reduction:
     - Applied Principal Component Analysis (PCA)
     - Reduced 14 features → 8 principal components
     - Preserved 75% cumulative variance

  3. Model Development:
     - Support Vector Machine (SVM)
     - Linear & RBF kernels evaluated
     - Hyperparameter tuning using GridSearchCV

Model Performance:
  | Metric    | Class 0 | Class 1 |
  | --------- | ------- | ------- |
  | Precision | 0.91    | 0.94    |
  | Recall    | 0.94    | 0.90    |
  | F1 Score  | 0.93    | 0.92    |

Visualizations Included:
  1. Scree Plot
  2. PCA Scatter Plot
  3. Biplot
  4. Cumulative Variance Plot 
  5. Confusion Matrix
  6. ROC Curve
  7. Precision-Recall Curve
  8. Decision Boundary Comparison

Tech Stack:
  - Python
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
  

