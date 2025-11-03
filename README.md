🧠 Decision Tree 

📘 Overview

This project implements a Decision Tree classifier from scratch and evaluates its performance on real-world data.
Through this assignment, I aimed to understand how decision trees split data, measure uncertainty, and balance between model complexity and generalization.

⸻

📊 Dataset
	•	Name: Breast Cancer Wisconsin (Diagnostic)
	•	Source: UCI Machine Learning Repository￼
	•	Samples: 569
	•	Features: 30 numeric features
	•	Target:
	•	0: Malignant
	•	1: Benign
  
⚙️ Implementation Details
Step
Description
1. Data Preprocessing
Standardized numeric features and encoded labels.
2. Splitting Criteria
Used Information Gain based on Entropy.
3. Tree Construction
Recursive binary splitting until purity or max depth.
4. Stopping Conditions
Minimum samples per node and maximum depth thresholds.
5. Evaluation
Calculated accuracy, confusion matrix, and visualized decision boundaries.

📈 Results
Metric
Training
Test
Accuracy
0.97
0.94
Depth
5
—
	•	The model performs well without significant overfitting.
	•	The decision boundaries are interpretable and align with feature importance.
	•	Compared to logistic regression, the tree offers higher interpretability.

⸻
🧩 Discussion
	•	Strengths:
	•	Simple and interpretable structure.
	•	Handles nonlinear relationships automatically.
	•	Weaknesses:
	•	Sensitive to noise; prone to overfitting without pruning.
	•	Small feature changes can lead to different tree structures.
	•	Improvement Ideas:
	•	Apply pruning or ensemble methods (e.g., Bagging or Random Forest).
	•	Experiment with Gini Index as an alternative splitting criterion.

⸻
🧮 Key Equations

Entropy = - \sum_i p_i \log_2(p_i)

Information\ Gain = H(parent) - \sum_{children} \frac{n_{child}}{n_{parent}} H(child)
