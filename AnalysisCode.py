#Code made with love using kk and gpt for its ability to know everything a package can do
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Loads the dataset using a bamboo eating panda. yes it actually is no cap frfr
df = pd.read_excel('HousePrice.xlsx')

#Displaying the basic dataset information
print("Dataset Overview:")
print(df.info())
print(df.describe())

#Checking for missing values cause its cool
print("Missing values:")
print(df.isnull().sum())

sns.pairplot(df, diag_kind='kde')
plt.show()

#Computing the correlation matrix graph
df['Garage'] = df['Garage'].map({'Yes': 1, 'No': 0})
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Defining some variables
X1 = df[['HouseSize']]
y = df['HousePrice']

# Model I      Simple Linear Regression
X1 = sm.add_constant(X1)  # Adding intercept
model1 = sm.OLS(y, X1).fit()
print("Model I Summary:")
print(model1.summary())

# Model II     Multiple Linear Regression (House Size + Bedrooms)
X2 = df[['HouseSize', 'Bedrooms']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2).fit()
print("Model II Summary:")
print(model2.summary())

#Model III    Multiple Regression (House Size + Bedrooms + Bathrooms + Garage)
X3 = df[['HouseSize', 'Bedrooms', 'Bathrooms', 'Garage']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(y, X3).fit()
print("Model III Summary:")
print(model3.summary())



def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    return mse, rmse, r2

models = {"Model I": (model1, X1), "Model II": (model2, X2), "Model III": (model3, X3)}
results = {}

for name, (model, X) in models.items():
    mse, rmse, r2 = evaluate_model(model, X, y)
    results[name] = {"MSE": mse, "RMSE": rmse, "R^2": r2}

results_df = pd.DataFrame(results).T
print("Model Evaluation:")
print(results_df)

#Graphs and shiz

residuals = model3.resid
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20)
plt.title("Residual Distribution (Model III)")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("QQ Plot of Residuals")
plt.show()

plt.scatter(model3.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

print("Analysis complete. The best model is selected based on R^2 and residual checks.")
