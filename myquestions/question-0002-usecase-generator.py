import pandas as pd
import numpy as np

def generar_caso_de_uso_imputar_por_grupo():

    grupos = ["A", "B", "C"]
    n_rows = 20

    df = pd.DataFrame({
        "grupo": np.random.choice(grupos, n_rows),
        "valor": np.random.uniform(10, 100, n_rows)
    })

    df.loc[np.random.choice(df.index, 5, replace=False), "valor"] = np.nan

    input_data = {
        "df": df.copy(),
        "grupo_col": "grupo",
        "valor_col": "valor"
    }

    df_out = df.copy()
    mediana_global = df_out["valor"].median()

    df_out["valor"] = df_out.groupby("grupo")["valor"].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else mediana_global)
    )

    return input_data, df_out
