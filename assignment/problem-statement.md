# Data Science Assignment
## Customer Churn Prediction

### Business Problem

You are a Data Scientist at a telecommunications company. The company wants to identify customers who are likely to churn so that the retention team can proactively engage with them.

Using the provided Telco Customer Churn dataset, build an end-to-end machine learning solution to predict customer churn.

**Dataset:** IBM Telco Customer Churn Dataset  
**Target Variable:** Churn (Yes / No)

## 1. Data Understanding & Preparation

Load and investigate the dataset.

Perform appropriate:

- Data type and structure checks
- Missing-value analysis
- Duplicate analysis
- Numerical and categorical feature identification
- Target-variable analysis
- Data cleaning and preprocessing
- Encoding of categorical variables
- Train/Test Split: Use a 70:30 split for training and testing.
- Random Seed: Set random_state = 42 for reproducibility.

Document important observations and explain key preprocessing decisions.

**Important:** Avoid data leakage and ensure that preprocessing can also be applied consistently to new/unseen data.

## 2. Exploratory Data Analysis

Perform EDA to understand customer behaviour and factors associated with churn.

Create at least 5 meaningful visualizations, covering appropriate numerical and categorical variables.

Your analysis should include:

- Churn distribution
- Customer/service characteristics
- Churn against important customer attributes
- Numerical variable distributions/relationships

For each major visualization, provide a brief business insight.

## 3. Feature Engineering

Create at least 2 meaningful features that could potentially improve churn prediction.

For each feature, explain:

- How it was created
- Why it may be useful

## 4. Model Development

Build a Decision Tree Classifier to predict Churn.

You must:

- Train the model on the training dataset.
- Experiment with at least two different Decision Tree configurations.
- Compare the models.
- Select and justify the final model.

## 5. Model Evaluation

Evaluate the final model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Interpret the results from a business perspective.

Also explain:

For a telecom company trying to identify customers who may churn, would you prioritize Precision or Recall? Why?

## 6. Model Interpretation

Explain what drives the model’s churn predictions.

Include:

- Feature importance
- Top features influencing churn
- Decision Tree visualization or appropriate interpretation

Provide a short explanation of the key findings.

## 7. Model Saving & API

Save the final model/preprocessing pipeline so that it can be reused for new customer data.

Build a REST API using Flask or FastAPI.

### Required Endpoint

**POST /predict**

The endpoint should:

1. Accept customer information as JSON.
2. Apply the required preprocessing.
3. Load/use the trained model.
4. Return the churn prediction.
5. Return churn probability.
6. Handle invalid input appropriately.

Example response:

```json
{
 "prediction": "Yes",
 "churn_probability": 0.82
}
```

## 8. Submission Requirements

Submit:

- Jupyter Notebook — complete analysis and modelling
- Python/API code
- Saved model/pipeline
- requirements.txt
- README.md with setup and execution instructions
- Sample API request and response

### Suggested structure

```
customer_churn_project/
│
├── data/
├── notebook/
│ └── churn_analysis.ipynb
├── model/
│ └── churn_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Expected Outcome

Your final submission should demonstrate the complete workflow:

**Business Problem → Data → Preparation → EDA → Feature Engineering → Model → Evaluation → Interpretation → Saved Model → API**

**Note:** Bonus marks may be awarded for additional activities such as hyperparameter tuning, handling class imbalance, comparing additional models, or other meaningful improvements to the solution.
