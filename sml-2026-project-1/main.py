from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # ─────────────────────────────────────────
    # 1. TRAIN / VALIDATION SPLIT  (80% / 20%)
    # ─────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        images, distances,
        test_size=0.20,
        random_state=42
    )
    print(f"[INFO]: Train: {len(X_train)} | Val: {len(X_val)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("knn", KNeighborsRegressor()),
    ])

    param_grid = {
        "pca__n_components": [0.90, 0.95],
        "knn__n_neighbors": [3, 5, 7],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("[INFO]: Starte GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_mae = -grid_search.best_score_
    print(f"[INFO]: Beste Parameter: {grid_search.best_params_}")
    print(f"[INFO]: Beste GridSearch MAE (CV): {round(best_mae, 3)}")

    val_pred = best_model.predict(X_val)
    print_results(y_val, val_pred, label="Validation")

    test_images = load_test_dataset(config)
    test_pred = best_model.predict(test_images)

    save_results(test_pred)
    print("[INFO]: Vorhersagen gespeichert in predictions.csv")
