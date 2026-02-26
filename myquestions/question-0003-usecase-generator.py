import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def generar_caso_de_uso_codificar_y_dividir():

    n_rows = 40

    df = pd.DataFrame({
        "cat": np.random.choice(["rojo", "azul"], n_rows),
        "num": np.random.randn(n_rows),
        "target": np.random.randint(0, 2, n_rows)
    })

    input_data = {
        "df": df.copy(),
        "target_col": "target",
        "test_size": 0.25
    }

    X = df.drop(columns=["target"])
    y = df["target"].values

    X_num = X.select_dtypes(include=[np.number]).values

    ohe = OneHotEncoder(sparse=False)
    X_cat = ohe.fit_transform(X.select_dtypes(include=["object"]))

    X_final = np.hstack([X_cat, X_num])

    result = train_test_split(
        X_final,
        y,
        test_size=0.25,
        random_state=42
    )

    return input_data, tuple(result)
