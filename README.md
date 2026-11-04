Mushroom Classification Assignment
Problem Statement
The objective of this project is to classify mushrooms as edible or poisonous using Machine Learning classification algorithms and compare their performance using multiple evaluation metrics.

Dataset Description
Dataset Name: Mushroom Dataset

Source: https://archive.ics.uci.edu/dataset/73/mushroom

Number of Instances: 8124

Number of Features: 22

Target Variable: Class (Edible / Poisonous)

GitHub Repository Link
Repository: https://github.com/isha1726/mushroom-classification-assignment

Streamlit Application Link
Application: https://mushroom-classification-assignment-nbvqt9xzxb6x55gmmrfspd.streamlit.app/

Models Used
Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Naive Bayes
Support Vector Machine (SVM)
Evaluation Metrics
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
Model Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.954462	0.983361	0.949239	0.956522	0.952866	0.908845
Decision Tree	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
KNN	0.995077	0.999979	0.989873	1.000000	0.994911	0.990192
Naive Bayes	0.902769	0.953738	0.895939	0.902813	0.899363	0.805340
Support Vector Machine	0.991385	0.999853	0.996124	0.985934	0.991003	0.982786

Observations on Model Performance
ML Model Name	Observation about Model Performance
Logistic Regression	Achieved 95.45% accuracy and provided a strong baseline classifier for the dataset.
Decision Tree	Achieved perfect performance with 100% Accuracy, AUC, Precision, Recall, F1 Score, and MCC. It was the best-performing model.
KNN	Achieved 99.51% accuracy and performed exceptionally well, close to Decision Tree.
Naive Bayes	Achieved 90.28% accuracy, which was the lowest among all models due to its feature independence assumption.
Support Vector Machine	Achieved 99.14% accuracy with excellent Precision, Recall, AUC, and MCC scores.
Overall Winner for Dataset	Decision Tree Classifier achieved the highest performance across all evaluation metrics and is selected as the best model.

Files Included
mushroom_classification.ipynb
app.py
test_data.csv
decision_tree.pkl
logistic.pkl
knn.pkl
naive_bayes.pkl
svm.pkl
requirements.txt
README.md

Conclusion
Five Machine Learning classification algorithms were implemented and evaluated on the Mushroom Dataset. The models were compared using Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC Score. Among all models, the Decision Tree Classifier achieved perfect scores across all evaluation metrics and was selected as the best-performing model for mushroom classification.
