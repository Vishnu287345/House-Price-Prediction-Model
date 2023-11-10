# Let's start by loading the dataset to examine its structure and contents
import pandas as pd

# Load the data from the CSV file
#file_path = "C:\Users\vishn\OneDrive\Desktop\intrn\pd2.csv"
data = pd.read_csv(r'C:\Users\vishn\OneDrive\Desktop\intrn\pd2.csv')

# Display the first few rows of the dataframe to understand what the data looks like
data.head()
# Checking the data types, number of unique values, and presence of null values in each column
data_info = pd.DataFrame({
    'Data Type': data.dtypes,
    'Unique Count': data.nunique(),
    'Null Count': data.isnull().sum()
})

# Display summary statistics of the numerical columns to get an idea of the distribution
numerical_summary = data.describe()

data_info, numerical_summary
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Dropping the 'Id' column as it is just an identifier
data.drop('Id', axis=1, inplace=True)

# Imputing missing values for numerical features with the mean
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
imputer_num = SimpleImputer(strategy='mean')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

# Encoding categorical variables
cat_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Normalizing numerical variables to a range of 0 to 1
scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Let's look at the transformed data
data.head()
import seaborn as sns
import matplotlib.pyplot as plt

# EDA: Let's visualize the distribution of the target variable 'SalePrice'
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Now, let's check the correlation of the numerical features with the target variable
# Compute the correlation matrix
corr_matrix = data.corr()

# Sort the correlations with SalePrice
sorted_correlations = corr_matrix['SalePrice'].sort_values(ascending=False)

# Let's look at the top 10 positive correlations and top 5 negative correlations
top_positive_correlations = sorted_correlations.head(11)[1:]  # excluding SalePrice itself
top_negative_correlations = sorted_correlations.tail(5)

# Visualize the top positive correlations
plt.figure(figsize=(10, 6))
top_positive_correlations.plot(kind='bar')
plt.title('Top 10 Positive Correlations with Sale Price')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()

# Visualize the top negative correlations
plt.figure(figsize=(10, 6))
top_negative_correlations.plot(kind='bar')
plt.title('Top 5 Negative Correlations with Sale Price')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()

# Display the correlation values
top_positive_correlations, top_negative_correlations
# Feature Engineering: Create new features based on the insights from EDA

# Age of the house at the time of the sale
data['HouseAge'] = data['YrSold'] - data['YearBuilt']

# Number of years since last remodeling when the house was sold
data['YearsSinceRemodel'] = data['YrSold'] - data['YearRemodAdd']

# Total square footage of the house (sum of basement, 1st and 2nd floor square footage)
data['TotalSqFt'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

# Normalizing the new features
data[['HouseAge', 'YearsSinceRemodel', 'TotalSqFt']] = scaler.fit_transform(data[['HouseAge', 'YearsSinceRemodel', 'TotalSqFt']])

# Now let's prepare the data for modeling
from sklearn.model_selection import train_test_split

# Define the features (X) and the target (y)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's check the shape of the resulting datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Function to train and evaluate models
def train_and_evaluate(models, X_train, y_train):
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # Calculate RMSE
        rmse = np.sqrt(-cv_scores.mean())
        print(f'{name} model - Average RMSE: {rmse}')

# Train and evaluate models
train_and_evaluate(models, X_train, y_train)
# To investigate the issue with the Linear Regression model, let's first check for any potential outliers in the target variable 'SalePrice'

# Plotting a boxplot to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data['SalePrice'])
plt.title('Boxplot of Sale Prices')
plt.xlabel('Sale Price')
plt.show()

# We will also check the condition number to assess multicollinearity, which can affect Linear Regression significantly
from numpy.linalg import cond

# Calculate the condition number of the training set
condition_number = cond(X_train)

condition_number
from sklearn.linear_model import Ridge
# Remove records with non-positive SalePrice values
y_train_cleaned = y_train[y_train > 0]
X_train_cleaned = X_train.loc[y_train_cleaned.index]

# Apply the log transformation to the cleaned target variable
y_train_log_cleaned = np.log(y_train_cleaned)

# Initialize and train the Ridge Regression model again
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_cleaned, y_train_log_cleaned)

# Evaluate the model using cross-validation on the cleaned data
cv_scores_ridge_cleaned = cross_val_score(ridge_model, X_train_cleaned, y_train_log_cleaned, cv=5, scoring='neg_mean_squared_error')

# Calculate RMSE for the Ridge model on the cleaned data
rmse_ridge_cleaned = np.sqrt(-cv_scores_ridge_cleaned.mean())
print(f'Ridge Regression model - Average RMSE (log-transformed target, cleaned data): {rmse_ridge_cleaned}')

# Re-evaluate the Linear Regression model with the log-transformed and cleaned target
linear_model = LinearRegression()
cv_scores_linear_cleaned = cross_val_score(linear_model, X_train_cleaned, y_train_log_cleaned, cv=5, scoring='neg_mean_squared_error')
rmse_linear_cleaned = np.sqrt(-cv_scores_linear_cleaned.mean())
print(f'Linear Regression model - Average RMSE (log-transformed target, cleaned data): {rmse_linear_cleaned}')

# Update the models dictionary with the retrained models on cleaned data
models['Ridge Regression'] = ridge_model
models['Linear Regression'] = linear_model  # Replacing the old Linear Regression model with the new one trained on log target
from sklearn.metrics import mean_squared_error

# Predicting on the test set with Ridge Regression and calculating RMSE
ridge_predictions_log = ridge_model.predict(X_test)
ridge_predictions = np.exp(ridge_predictions_log)  # Revert the log transformation with the exponential function
ridge_rmse_test = np.sqrt(mean_squared_error(y_test, ridge_predictions))
print(f'Ridge Regression model - RMSE on test set: {ridge_rmse_test}')

# Predicting on the test set with Random Forest and Gradient Boosting models
for name, model in models.items():
    if name not in ['Ridge Regression', 'Linear Regression']:  # We've already evaluated Ridge Regression
        # Predict and calculate RMSE
        predictions = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, predictions))
        print(f'{name} model - RMSE on test set: {rmse_test}')

# Since the Linear Regression model had issues, we'll exclude it from the test evaluation
# If needed, we could revisit and apply robust regression or further diagnostics to understand its performance issues
