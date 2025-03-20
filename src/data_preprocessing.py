import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the cardiovascular dataset

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame:
        - Processed DataFrame with age converted from days to years
        - Outliers dropped from ap_hi and ap_lo
        - Data types converted to category
        - Drop redundant columns
    """
    print("Load data...")
    df = pd.read_csv(file_path, delimiter=";")

    print("Converting age from days to years...")
    # Convert age from days to years
    df["age"] = (df["age"] / 365).round().astype(int)

    # Remove outliers
    print("Removing Outliers...")
    remove_outliers_ap_hi = df.loc[
        (df['ap_hi'] <= 250) & (df['ap_lo'] >= 25) &
        (df['ap_lo'] <= 200) & (df['ap_lo'] >= 20)
    ]

    # Convert categorical data types into category
    print("Converting data to categorical type...")
    categorical_col = ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    df[categorical_col] = df[categorical_col].astype("category")

    # Create BMI column
    print("Creating new colum called BMI...")
    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)

    # Drop Columns
    print("Dropping redundant columns...")
    df.drop(columns=['id', 'weight', 'height'], inplace=True)

    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets

    Parameters:
    pd.DataFrame

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test
    """
    print("Splitting data...")
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    return train_test_split(X, y, test_size=0.2, random_state=19)


if __name__ == "__main__":
    """
    Save splitted training and testing files into '../data/"
    """
    df = load_data("data/cardio_train.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
