🚀 Crowdfunding Campaign Success Prediction
A machine learning project designed to analyze crowdfunding dataset attributes and predict whether a campaign will be successful (IsSuccessful = 1) or not (IsSuccessful = 0). The project involves data preprocessing, feature selection, feature scaling, model training, hyperparameter tuning, and performance evaluation using multiple classification algorithms.

📌 Project Overview
Predicting campaign success helps platform owners and creators optimize fundraising strategies. In this project, multiple classification models were trained and cross-validated to identify the best algorithm for predicting success based on key features like goal amount, raised amount, campaign duration, owner experience, and media presence.

Key Highlights:
Dataset Size: 100,000 rows × 13 initial columns
Selected Best Model: Support Vector Machine (SVM with Linear Kernel)
Top Model Performance: 99.96% Accuracy (F1-Score: 0.9996)
🛠️ Tech Stack & Libraries
Language: Python 3.x
Data Processing & Analysis: Pandas, NumPy
Data Visualization: Matplotlib, Seaborn
Machine Learning & Pipeline: Scikit-Learn
Preprocessing & Feature Selection: LabelEncoder, StandardScaler, SelectKBest (f_classif)
Models: RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, GaussianNB, SVC
Evaluation & Tuning: GridSearchCV, cross_val_score, accuracy_score, classification_report, confusion_matrix
📊 Dataset & Feature Selection
The original dataset contained campaign metadata. After initial cleaning, non-informative identifier columns (CampaignID, Unnamed: 0) were removed.

Feature Selection (SelectKBest)
Using f_classif ANOVA F-test, the top 6 most influential features were selected for model training:

GoalAmount
RaisedAmount
DurationDays
OwnerExperience
VideoIncluded
NumUpdates
📈 Model Performance & Comparison
All models were evaluated after hyperparameter optimization using GridSearchCV:

Model	Accuracy	Precision	Recall	F1-Score
Support Vector Machine (SVM)	0.9996	0.9996	0.9996	0.9996
Random Forest	0.9558	0.9581	0.9558	0.9557
Decision Tree	0.9297	0.9343	0.9297	0.9295
K-Nearest Neighbors (KNN)	0.8957	0.8958	0.8957	0.8957
Gaussian Naive Bayes	0.6622	0.6782	0.6622	0.6549
🔍 Key Insights & Confusion Matrix Summary
SVM (Best Performer): Achieved near-perfect classification with only 17 misclassifications out of 20,000 test instances (C=10, kernel='linear').
Random Forest & Decision Tree: Demonstrated high accuracy and non-linear feature handling.
KNN: Showed strong performance after standard scaling with Manhattan distance metric (k=11).
Naive Bayes: Underperformed compared to tree/boundary models due to feature correlation and linear dependencies.
📄 Final Conclusion SVM (Linear Kernel) was selected as the final production model due to its exceptional accuracy, precision, and balanced recall across all classes.
