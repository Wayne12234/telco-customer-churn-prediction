# Telco Customer Churn Prediction

A beginner-friendly Machine Learning project that predicts whether a telecommunications customer is likely to leave a company using Python and Logistic Regression.

## Project Overview

Customer churn is a major challenge for telecommunications companies. Losing customers affects revenue and business growth.

The objective of this project is to build a classification model that identifies customers who are likely to discontinue their services based on their historical information.

By identifying potentially high-risk customers, businesses can take proactive steps to improve customer retention.

## Dataset

The project uses a Telco Customer Churn dataset containing 7,043 customer records.

The dataset includes information such as:

- Customer tenure
- Monthly charges
- Total charges
- Contract type
- Internet service
- Payment method
- Technical support
- Customer churn status

**Target Variable:** `Churn`

- Yes (1): Customer left the company.
- No (0): Customer stayed with the company.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Project Workflow

1. **Data Loading:** Imported the dataset using Pandas.
2. **Data Exploration:** Examined dataset structure, columns, data types and customer distribution.
3. **Data Cleaning:** Converted TotalCharges into numerical format and handled missing values.
4. **Feature Selection:** Separated independent variables (X) and the target variable (y).
5. **Train-Test Split:** Divided the dataset into 80% training and 20% testing data.
6. **Data Preprocessing:**
   - StandardScaler for numerical features.
   - OneHotEncoder for categorical features.
7. **Model Training:** Implemented Logistic Regression using Scikit-learn.
8. **Model Evaluation:** Evaluated predictions using accuracy, precision, recall, F1-score and a confusion matrix.

## Model Performance

Initial Logistic Regression results:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Churn Precision | 0.66 |
| Churn Recall | 0.56 |
| Churn F1-score | 0.60 |

The model achieved approximately 80.55% accuracy on the test dataset.

However, the 56% recall for churned customers indicates that the model still misses a considerable number of customers who actually leave.

This highlights an opportunity for further improvement.

## Business Objective

The model aims to help telecommunications companies:

- Identify customers who may be at risk of leaving.
- Prioritize customers for retention campaigns.
- Understand patterns associated with customer churn.
- Make more informed customer retention decisions.

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/Wayne12234/telco-customer-churn-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd telco-customer-churn-prediction
   ```

3. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn
   ```

4. Run the Python script:

   ```bash
   python Telco.py
   ```

## Future Improvements

- Improve churn recall.
- Experiment with other classification algorithms.
- Perform feature importance analysis.
- Tune classification thresholds.
- Explore customer churn patterns in greater detail.

## Key Learnings

This project helped me understand the fundamentals of supervised Machine Learning, including data preprocessing, classification, model training, predictions and performance evaluation.

It is part of my ongoing journey of learning Artificial Intelligence and Machine Learning through practical projects.

---

**Project Status:** Completed — Initial Version

**Author:** Wayne12234
