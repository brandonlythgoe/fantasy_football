import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Load your dataset
football = pd.read_csv('fantasy_merged_7_17.csv')

# Drop unnecessary columns (if any)
football = football.drop(['Player', 'PlayerID', 'Tm', 'Rk', 'PPR', 'Player', 'Year', 'GS'], axis=1)

# One-hot encode categorical variables
football_encoded = pd.get_dummies(football, columns=['FantPos'])

# Split the data into features (X) and target (y)
X = football_encoded.drop('PosRk', axis=1)
y = football_encoded['PosRk']

# Split the data into training and holdout sets
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=52)

# Define a simpler parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [25, 50, 75, 100, 125, 150, 175],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.1, 0.3, 0.5, 0.7, 1.0],
    'subsample': [0.3, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'lambda': [0, 1, 2, 3, 4, 5],
    'alpha': [0, 1, 2, 3, 4, 5]
}

# Initialize XGBoost Regressor
xgb = XGBRegressor()

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, n_iter=30,
                                   scoring='neg_mean_squared_error', cv=3, random_state=52, n_jobs=-1)

random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Create the XGBoost model with the best hyperparameters
best_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    min_child_weight=best_params['min_child_weight'],
)

# Fit the model on the training data
best_model.fit(X_train, y_train)

# Streamlit app
st.title('XGBoost Model Interface')

# Form to input user values
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"{feature}:", value=0.0, step=0.1)

# Predictions and display
if st.button('Predict'):
    # Create a DataFrame from user input
    user_df = pd.DataFrame([user_input])

    # Ensure the user input has the same columns as the training data
    missing_columns = set(X.columns) - set(user_df.columns)
    for column in missing_columns:
        user_df[column] = 0

    # Reorder columns to match the training data
    user_df = user_df[X.columns]

    # Make predictions using the trained model
    user_prediction = best_model.predict(user_df)

    # Display the prediction to the user
    st.success(f"Predicted Position Rank : {user_prediction[0]}")

# Display model evaluation metrics
st.header('Model Evaluation Metrics on Holdout Set')
predictions_holdout = best_model.predict(X_holdout)
mse_holdout = mean_squared_error(y_holdout, predictions_holdout)
mae_holdout = mean_absolute_error(y_holdout, predictions_holdout)
r2_holdout = r2_score(y_holdout, predictions_holdout)

st.write(f'Mean Squared Error on Holdout Set: {mse_holdout}')
st.write(f'Mean Absolute Error on Holdout Set: {mae_holdout}')
st.write(f'R^2 Score on Holdout Set: {r2_holdout}')
