

# Intrusion Detection (Mini Project)



### A simple machine learning project for intrusion detection using the UNSW-NB15 dataset.



## Dataset



\- UNSW-NB15

\- Training and testing sets are used for evaluation.



## Models



\- Decision Tree

\- Random Forest

\- Random Forest with Pipeline + GridSearchCV



## Workflow



\- Load training and testing data

\- Preprocess features (one-hot encoding for categorical variables)

\- Train baseline models (Decision Tree, Random Forest)

\- Build pipeline and tune hyperparameters with GridSearchCV

\- Evaluate models on the test set


## Goal



The goal was to classify network traffic as either normal or attack.
- Started with baseline models (Decision Tree, Random Forest) to establish initial performance.
- Improved the workflow by building a Pipeline that handles preprocessing (one-hot encoding for categorical features) and integrates directly with the model.
- Applied GridSearchCV for hyperparameter tuning, optimizing Random Forest parameters such as depth, number of estimators, and class weights.
- This improvement boosted performance significantly, achieving 77% accuracy and an F1-score of 0.81, compared to ~53–68% with baseline models.




