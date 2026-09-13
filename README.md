**Loan Approval Prediction Using Machine Learning**
**Project Overview**
This project develops a machine learning classification model to predict loan approval decisions using applicant financial, demographic, employment, and credit-related information.
The project was developed for a FinTech lending business seeking to automate part of its loan screening process while reducing the financial risk associated with incorrect lending decisions.
The primary business concern is that the cost of an incorrect decision is not equal:
•	Incorrectly rejecting a creditworthy applicant: approximately $8,000 in lost profit
•	Incorrectly approving a high-risk applicant who defaults: approximately $50,000 in potential loss
Because the cost of an incorrect approval is significantly higher, the project places particular emphasis on precision, recall, ROC-AUC, confusion matrix analysis, and business-oriented threshold considerations, rather than relying on accuracy alone.

**Business Problem**
Manual loan assessment can be time-consuming and may produce inconsistent decisions when large numbers of applications need to be processed.
The objective of this project is to build a classification model that can:
1.	Predict whether a loan application should be approved or rejected.
2.	Identify the financial and applicant characteristics that most influence approval decisions.
3.	Minimize incorrect approvals, particularly approvals of applicants who are likely to default.
4.	Support loan officers by providing consistent, data-driven risk assessments.
5.	Provide insights that can improve the organization's lending strategy.

**Machine Learning Objective**
This is a binary classification problem.

The target variable is: LoanApproved

Where:
•	0 = Loan not approved
•	1 = Loan approved

The dataset contains 19,000 loan applications, with:
•	15,220 rejected applications
•	4,780 approved applications

This represents an imbalanced target distribution, making metrics such as precision, recall, F1-score, and ROC-AUC particularly useful for evaluating the model.

**Dataset Features**
The dataset contains a combination of numerical, categorical, ordinal, and binary variables.
Examples of important features include:

Financial Features
•	Annual Income
•	Monthly Income
•	Loan Amount
•	Monthly Debt Payments
•	Checking Account Balance
•	Savings Account Balance
•	Total Assets
•	Total Liabilities
•	Net Worth
•	Debt-to-Income Ratio

Credit and Risk Features
•	Credit Score
•	Previous Loan Defaults
•	Bankruptcy History
•	Payment History
•	Number of Credit Inquiries
•	Interest Rate
•	Base Interest Rate

Demographic and Employment Features
•	Age
•	Experience
•	Employment Status
•	Education Level
•	Marital Status
•	Home Ownership Status
Loan Information
•	Loan Purpose
•	Loan Amount
•	Interest Rate

**Project Workflow**
The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

1. Business Understanding
Defined the FinTech lending problem, business costs, objectives, and success criteria.

2. Data Understanding
Explored the dataset to understand:
•	Data types
•	Target distribution
•	Missing values
•	Numerical and categorical variables
•	Relationships between variables
•	Potential correlations and redundant features

3. Data Preparation
The preprocessing stage included:
•	Handling missing values
•	Converting incorrectly formatted numerical variables
•	Separating numerical and categorical features
•	Handling ordinal variables appropriately
•	Encoding categorical variables
•	Scaling numerical variables
•	Creating engineered features and interaction terms
•	Building reusable preprocessing pipelines

5. Modeling
Two classification algorithms were evaluated:
•	Logistic Regression
•	Random Forest
**Logistic Regression** was selected as the final model based on its stronger overall test performance and excellent ROC-AUC.

6. Evaluation
The models were evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	ROC-AUC
•	Confusion Matrix
•	ROC Curve
•	Segment-level performance analysis

7. Business Interpretation
The final model was assessed not only from a technical perspective but also according to the financial consequences of incorrect loan decisions.

**Data Preprocessing**
A ColumnTransformer and Scikit-learn pipelines were used to apply different preprocessing strategies to different types of variables.

Numerical Features
Numerical variables were processed using:
•	Median imputation for missing values
•	StandardScaler for scaling
Scaling was particularly important for Logistic Regression because the model is sensitive to differences in feature magnitude.

Categorical Features
Categorical variables were processed using:
•	Most-frequent imputation
•	One-hot encoding
•	drop='first' to reduce redundant dummy variables

Ordinal Features
Ordinal variables were transformed while preserving their natural ordering.

Binary Features
Binary variables were handled separately so that their existing 0/1 representation was preserved.
Using pipelines helped ensure that preprocessing was performed consistently during both model training and testing.

** Models Evaluated**
**Logistic Regression**
Logistic Regression was used because:
•	It is appropriate for binary classification.
•	It provides interpretable coefficients.
•	Its coefficients help explain which features influence approval decisions.
•	It performs well with standardized numerical features and encoded categorical variables.

**Random Forest**
Random Forest was evaluated as a non-linear alternative because it can capture complex relationships and interactions between variables without requiring the same assumptions as Logistic Regression.
After optimization and evaluation, Logistic Regression demonstrated the stronger overall performance.

**Final Model Performance**
The final Logistic Regression model achieved the following results on the unseen test dataset:
Metric	Score
Accuracy	94.85%
Precision	84.79%
Recall	95.61%
F1 Score	89.87%
ROC-AUC	99.13%
The test set contained 4,000 observations.

**Interpretation**
The model correctly classified approximately 95% of loan applications.
Its 95.61% recall indicates that it successfully identifies most applicants belonging to the positive class.
The 84.79% precision indicates that most applicants predicted as positive are correctly classified, although some incorrect positive predictions remain.
The 89.87% F1-score demonstrates a strong balance between precision and recall.
Most importantly, the 99.13% ROC-AUC indicates that the model has excellent ability to distinguish between the two loan decision classes.

**Classification Report**
Class	Precision	Recall	F1-score	Support
0	0.99	0.95	0.97	3,044
1	0.85	0.96	0.90	956
Overall Accuracy			0.95	4,000
The model performs strongly for both classes, although precision for Class 1 is lower than for Class 0.
Because the business objective is to minimize incorrectly approving high-risk applicants, the false-positive predictions for the approval class require particular attention.

**Confusion Matrix**
The final model produced the following test-set results:
	Predicted 0	Predicted 1
Actual 0	2,880	164
Actual 1	42	914
The model correctly classified:
•	2,880 Class 0 applications
•	914 Class 1 applications
It incorrectly classified:
•	164 Class 0 applications as Class 1
•	42 Class 1 applications as Class 0
From a business perspective, if Class 1 represents an approved loan, the 164 false positives represent applicants who were predicted for approval but actually belonged to the non-approved/risk class. These errors are particularly important because incorrect approvals carry a substantially higher financial cost.

**ROC-AUC Analysis**
The model achieved a ROC-AUC of 0.9913, indicating excellent classification performance.
The ROC curve is very close to the top-left corner, showing that the model can achieve a high true-positive rate while maintaining a relatively low false-positive rate across different classification thresholds.
This provides strong evidence that the model effectively separates the two loan decision classes.

 **Feature Importance and Model Interpretation**
Because Logistic Regression was selected as the final model, its coefficients were examined to understand which features had the strongest influence on predictions.
The analysis showed that some of the strongest predictors included:
•	Bankruptcy History
•	Previous Loan Defaults
•	Employment Status – Unemployed
•	Total Debt-to-Income Ratio
•	Credit Score interactions
•	Payment History
•	Loan Amount and Interest Rate interactions
•	Age-related interactions
•	Number of Credit Inquiries and Base Interest Rate interactions

**Key Finding**
Bankruptcy History and Previous Loan Defaults showed particularly strong negative coefficients, indicating that applicants with these characteristics were strongly associated with a lower probability of loan approval.
The model also identified unemployment and higher debt-related measures as important negative factors.
These results are consistent with the business logic that previous credit problems and high financial obligations can indicate increased lending risk.

**Business Recommendations**
The model's findings provide several potential recommendations for the lending business.

1. Strengthen checks for previous defaults and bankruptcy
Applicants with previous defaults or bankruptcy history should receive additional risk assessment because these variables were among the strongest negative predictors.

2. Monitor debt-to-income levels
High debt relative to income can indicate limited repayment capacity. The business could establish appropriate debt-to-income thresholds or additional review procedures for high-risk applications.

3. Consider employment stability
Employment status was an important predictor. Additional affordability checks could be considered for applicants without stable employment.

4. Introduce additional review for high-risk applications
   
Applications combining several risk indicators, such as:
•	Previous defaults
•	Bankruptcy history
•	High debt-to-income ratio
•	Unemployment
•	Weak payment history
could be flagged for additional human review.

5. Optimize the classification threshold
A default threshold of 0.50 may not necessarily be optimal for the business.
Since incorrectly approving a high-risk borrower has a much greater financial cost than rejecting a potentially good applicant, the business could consider a more conservative approval threshold.
This may increase precision and reduce risky approvals, although it could also increase false rejections.

6. Maintain human oversight
The model should support credit officers rather than completely replace human judgment. Borderline or high-risk cases can be reviewed manually before a final decision is made.

**Bias and Limitations**
Although the model performs strongly, several limitations should be considered.
Potential Bias
The model may perform differently across:
•	Income groups
•	Age groups
•	Credit-score categories
•	Employment statuses
•	Other applicant segments
Underrepresented groups may also receive less reliable predictions because the model has fewer examples from which to learn.
Historical loan decisions may contain existing human or institutional biases that the model could reproduce.

**Limitations**
The model:
•	Relies on historical data.
•	Cannot predict unexpected changes in an applicant's circumstances.
•	May experience reduced performance when economic conditions change.
•	May be affected by missing or incomplete information.
•	Is limited by the features available in the dataset.
•	Does not establish causal relationships between features and loan approval.
•	Requires ongoing monitoring to detect model or concept drift.
Therefore, strong test performance should not be interpreted as proof that the model will always perform equally well in real-world lending environments.

**Segment-Level Evaluation**
The model should also be evaluated across different customer segments to identify potential performance differences.
Recommended segments include:
•	Age groups
•	Income groups
•	Credit-score categories
•	Employment status
Precision is particularly important when evaluating these segments because the business objective is to reduce incorrect approvals.
A large difference in precision or false-positive rates between groups could indicate that the model requires further investigation or calibration.

**Technologies Used**
•	Python
•	Pandas
•	NumPy
•	Matplotlib
•	Scikit-learn
•	Jupyter Notebook / Visual Studio Code
•	Git & GitHub

**Machine Learning Techniques**
•	Data preprocessing
•	Feature engineering
•	ColumnTransformer
•	Pipeline
•	FeatureUnion
•	Logistic Regression
•	Random Forest
•	Cross-validation
•	GridSearchCV
•	Classification metrics
•	ROC-AUC analysis
•	Confusion matrix analysis
•	Feature coefficient interpretation
•	Segment-level evaluation

 **Repository Structure**
ML_LoanApproval
│
├── Banner_ML.png 
│
├── README.md
│
├── financial_loan_data.csv
│
├── financial_loan_risk.ipynb

**How to Run the Project**
1. Clone the repository
2. Navigate to the project directory
3. Install the required libraries
4. Run the Jupyter Notebook
Open the financial_loan_risk.ipynb and run the cells sequentially.

**Requirements**
pandas
numpy
matplotlib
scikit-learn
jupyter

**Key Project Outcome**
The project successfully developed and evaluated a machine learning solution for predicting loan approval decisions.
The final Logistic Regression model achieved 94.85% accuracy and a 99.13% ROC-AUC on the test dataset, demonstrating excellent predictive performance.
However, because the financial cost of incorrectly approving a high-risk applicant is significantly greater than the cost of rejecting a potentially good applicant, model deployment should prioritize controlling false-positive approvals rather than maximizing accuracy alone.
The project therefore recommends combining model predictions with threshold optimization, segment-level monitoring, risk-based review, and human oversight to support responsible lending decisions.

**Author**
Monica Nyagaya
Data Science Project – Loan Approval Prediction
Developed as part of a Data Science learning and portfolio project.

**License**
This project is intended for educational and portfolio purposes. The dataset, model, and results should not be used as the sole basis for real-world lending decisions without appropriate validation, governance, regulatory review, and fairness assessment.

