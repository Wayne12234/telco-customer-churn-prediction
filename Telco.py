
import pandas as pd
import numpy as np

df = pd.read_csv("telco.csv.csv")
# print(df)

# Basic dataset information
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.columns)

# Establish the business baseline

# How many customers are left and how many churned
# def custinfo():
#     return df[df["Churn"] == "Yes"] , df[df["Churn"] == "No"]
# churned, retained = custinfo()
# print("Churned customers : ", len(churned))
# print("Retained customers : ", len(retained))
# print("Churned ratio : ", round(len(churned)/len(df)*100 , 2), "%")

# print(df["TotalCharges"].dtype)
# num_val = pd.to_numeric(df["TotalCharges"], errors="coerce")
# print(num_val.dtype)
# print(num_val.isnull().sum())


# calculating confidence intervals
# p = (df["Churn"]=="Yes").mean()
# margin =np.sqrt(p*(1-p)/len(df)) * 1.96

# print("Lower limit : ", round((p-margin)*100 , 2),"%")
# print("Upper limit : ", round((p+margin)*100 , 2),"%")

# Does contract type affect churn ratio?
# from scipy.stats import chi2_contingency
# contract_churn = pd.crosstab(df["Contract"], df["Churn"])
# chi, dof, p_value, expected = chi2_contingency(contract_churn)

# print("Chi_square : ", round(chi,2))
# print("P-Values : ", p_value)
# print("Degrees of freedom : ", dof)
# print("Expected Values : ", expected)

# if p_value < 0.05:
#     print("Reject h0 : Contract type is associated with churn")
# else:
#     print("Contract type is not associated with churn")

# cont_type = pd.crosstab(df["Contract"] , df["Churn"] , normalize="index") * 100
# print(cont_type.round(2).astype(str) + "%") 

# Independent T testing

# from scipy.stats import ttest_ind
# churned = df[df["Churn"] == "Yes"]["MonthlyCharges"]
# retained = df[df["Churn"] == "No"]["MonthlyCharges"]

# t_stat, p_value =  ttest_ind(churned, retained, equal_var=False)
# print(" T stat : ",  round(t_stat,2))
# print("P-value : ", round(p_value,2))

# if p_value < 0.05:
#     print("Reject h0 : Monthly charges differ significantly")
# else:
#     print("Reject h0: no significant difference")

# ANOVA Testing
# monthly = df[df["Contract"] == "Month-to-month"]["MonthlyCharges"]
# yearly = df[df["Contract"] == "One year"]["MonthlyCharges"]
# two_year = df[df["Contract"] == "Two year"]["MonthlyCharges"]

# from scipy.stats import f_oneway

# f_stat, p_value = f_oneway(monthly, yearly, two_year)

# if p_value < 0.05:
#     print("Reject H0: At least one group mean is different")
# else:
#     print("Fail to reject H0")

# One sample testing

# monthly = df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].mean()
# yearly = df[df["Contract"] == "One year"]["MonthlyCharges"].mean()
# two_year = df[df["Contract"] == "Two year"]["MonthlyCharges"].mean()

# from scipy.stats import ttest_1samp
# t_stat , p_value = ttest_1samp(df["MonthlyCharges"] , 65)
# print(f"T stat {t_stat} : , P-value : {p_value}")

# if p_value < 0.05:
#     print("Reject H0: Average monthly charges differ significantly from $65")
# else:
#     print("Fail to reject H0: No significant difference from $65")

# Co-relation and regression analysis

num_val = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# correlation = df["tenure"].corr(num_val)
# print(round(correlation))

# co-relation matrix

# corr_matrix = df[["tenure","MonthlyCharges"]].corr()
# print(corr_matrix)

# Linear regression

# from sklearn.linear_model import LinearRegression
# X = df[["tenure"]]
# Y =num_val

# model = LinearRegression()
# model.fit(X,Y)

# predictions = model.predict(X)

# print("Slope : ", model.coef_[0])
# print("Intercept : ", model.intercept_)
# print("1st 5 predictions : ", np.round(predictions[:5],2))

# Model training
df["TotalCharges"] = num_val
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["customerID" , "Churn"])
y = df["Churn"].map({"Yes" : 1, "No" : 0})

print(X.shape)
print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2, random_state=42, stratify=y)

print("Testing data : ", X_test.shape)
print("Training data : ", X_train.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# Identifying numerical and categorail data
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

# Preprocess the columns
preprocessor = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
)

# Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 7. Evaluate model performance
print("Accuracy:", np.round(accuracy_score(y_test, y_pred)))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Check the first 10 test customers
results = X_test.head(10).copy()

# Add actual and predicted churn values
results["Actual Churn"] = y_test.head(10)
results["Predicted Churn"] = model.predict(X_test.head(10))

print(results[["Actual Churn", "Predicted Churn"]])
