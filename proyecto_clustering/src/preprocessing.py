import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path="data/mall_customers.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # Convertir género a variable numérica
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)


    return df, X_scaled