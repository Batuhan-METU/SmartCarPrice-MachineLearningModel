import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(path: str): 
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    df = df.drop_duplicates(keep = "first", ignore_index=True)

    df.loc[df["seats"] == 0, "seats"] = 5
    df = df[df["selling_price"] < 15000000]
    df = df[df["km_driven"]<1000000]

    return df

def preprocess_data(df: pd.DataFrame):
    X = df.drop("selling_price", axis = 1)
    y = df["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=15)

    cat_cols = df.select_dtypes("object").columns.to_list()
    onehot_columns = ["seller_type", "fuel_type", "transmission_type"]
    freq_columns = ["car_name", "brand", "model"]

    for col in freq_columns: 
        freq = X_train[col].value_counts() / len(X_train)

        X_train[col + "_freq"] = X_train[col].map(freq)
        X_test[col + "_freq"] = X_test[col].map(freq)

        mean_freq = freq.mean()
        X_test[col + "_freq"] = X_test[col + "_freq"].fillna(mean_freq)

    X_train = X_train.drop(["car_name", "brand", "model"], axis = 1)
    X_test = X_test.drop(["car_name", "brand", "model"], axis = 1)

    transformer = ColumnTransformer(
    transformers = [
        ("onehot", OneHotEncoder(drop = "first", handle_unknown="ignore"), onehot_columns)
    ], remainder="passthrough"
    )

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    encoded_columns = transformer.get_feature_names_out()

    X_train = pd.DataFrame(X_train, columns = encoded_columns)
    X_test = pd.DataFrame(X_test, columns= encoded_columns)

    return X_train, X_test, y_train, y_test, transformer, encoded_columns

