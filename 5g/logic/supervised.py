import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_regression(df, target_col):
    df = df.dropna()  # drop rows with missing values
    
    # Separate X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert categorical columns to numeric (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Ensure y is numeric (if not, convert)
    if y.dtype == 'O':
        y = pd.factorize(y)[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "predictions": y_pred,
        "y_test": y_test,
        "columns": X.columns.tolist()
    }
