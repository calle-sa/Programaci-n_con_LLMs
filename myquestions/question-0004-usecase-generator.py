import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def generar_caso_de_uso_evaluar_clasificador():

    n_rows = 30

    X = np.random.rand(n_rows, 5)
    y = np.random.randint(0, 2, n_rows)
    k = 3

    input_data = {
        "X": X,
        "y": y,
        "n_neighbors": k
    }

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="accuracy"
    )

    return input_data, float(np.mean(scores))
