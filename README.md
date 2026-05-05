# AirBnB Rental Price Prediction

## Project Overview
This project aims to predict AirBnB rental prices based on various features using a linear regression model. The insights gained from this predictive model can be beneficial for AirBnB hosts to set competitive prices, for guests to find the best deals, and for real estate investors and city planners to understand the housing market better.

## Data
The dataset used for this analysis contains information about AirBnB listings, including geographical location, room type, pricing, and review details. The initial dataset was loaded from `/content/drive/MyDrive/(2) Tuesday 8:30pm - Ziyan/Workfile/dataSP23 (1).csv`.

### Initial Data Exploration
- Loaded data into a pandas DataFrame.
- Displayed the first few rows of the DataFrame (`df.head()`).
- Checked data types and non-null counts using `df.info()`.

## Data Preprocessing

1.  **Column Dropping:** Irrelevant columns such as `id`, `name`, `host_id`, `host_name`, `neighbourhood`, `last_review`, `reviews_per_month`, and `calculated_host_listings_count` were dropped to simplify the model and remove features that might not be directly impactful or contain too many missing values.

2.  **Feature Engineering:** A new feature, `total_price`, was created by multiplying `price` and `minimum_nights` to represent the total cost for the minimum stay. The original `price` and `minimum_nights` columns were then dropped.

3.  **Missing Values:** Checked for missing values using `df.isnull().sum()`. The dataset was found to have no missing values after the initial cleaning steps.

4.  **Outlier Detection:** A custom function `check_outliers` was defined to identify outliers based on the Interquartile Range (IQR) method. This was applied to the `total_price` column.

5.  **Categorical Encoding:** Categorical features `neighbourhood_group` and `room_type` were converted into numerical representations using `LabelEncoder` from `sklearn.preprocessing`.

## Model Training

1.  **Splitting Data:** The dataset was split into features (`X`) and the target variable (`y`, which is `total_price`).

2.  **Train-Test Split:** The data was further divided into training and testing sets using `train_test_split` with a `test_size` of 20% and `random_state=42`.

3.  **Feature Scaling:** `StandardScaler` was used to scale the features in both the training and testing sets to standardize their range, which is crucial for many machine learning algorithms.

4.  **Linear Regression Model:** A `LinearRegression` model from `sklearn.linear_model` was initialized and trained using the scaled training data (`X_train`, `y_train`).

## Model Evaluation

1.  **Prediction:** The trained model was used to make predictions on the scaled test set (`X_test`).

2.  **R2 Score:** The R2 score was calculated to evaluate the model's performance, indicating how well the model explains the variability of the dependent variable. The R2 score achieved was `-18.06`, suggesting a very poor fit.

3.  **Cross-Validation:** Cross-validation was performed using `cross_val_score` with `cv=5` (and later with `KFold`) to get a more robust estimate of the model's performance. The average cross-validation R2 score was around `0.03` to `0.036`, further confirming the model's limited predictive power.

## Visualizations

Several plots were generated to visualize the data and potential relationships:

1.  **Rug Plot for `total_price`:** `sns.rugplot(df['total_price'])` was used to visualize the distribution of `total_price`.

2.  **Descriptive Statistics:** `df['total_price'].describe()` provided statistical summaries of the `total_price` column.

3.  **Scatter Plots:** 
    -   `df.plot(kind='scatter', x='neighbourhood_group', y='total_price')`
    -   `df.plot(kind='scatter', x='room_type', y='total_price')`
    These plots helped to visually inspect the relationship between categorical features and the target variable.
