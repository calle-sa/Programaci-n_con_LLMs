import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_preparar_sensores():

    n_rows = np.random.randint(15, 30)
    n_feats = np.random.randint(2, 4)

    data = np.random.randn(n_rows, n_feats) * 10
    df = pd.DataFrame(data, columns=[f"s{i}" for i in range(n_feats)])

    target_col = "target"
    df[target_col] = np.random.randint(0, 2, n_rows)

    input_data = {"df": df.copy(), "target_col": target_col}

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    lower = X.quantile(0.05)
    upper = X.quantile(0.95)

    X_clipped = X.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped)

    return input_data, (X_scaled, y)
