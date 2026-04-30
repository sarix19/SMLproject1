from utils import load_config, load_dataset, load_test_dataset, save_results

import numpy as np
import time

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":

    start_time = time.time()

    config = load_config()
    images, distances = load_dataset(config)

    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # -------------------------
    # Grid stays unchanged
    # -------------------------
    pca_variance = [0.90, 0.92, 0.94, 0.96, 0.98]
    n_neighbors_list = [1, 3, 5, 7, 9, 11, 15]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_mae = float("inf")
    best_params = None

    # -------------------------
    # Pre-scale ONCE
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(images)

    print("\n[INFO] Running optimized hyperparameter search...")

    for pca_var in pca_variance:

        # -------------------------
        # PCA ONCE per setting
        # -------------------------
        pca = PCA(n_components=pca_var, svd_solver="full", random_state=42)
        X_transformed = pca.fit_transform(X_scaled)

        for n_neighbors in n_neighbors_list:

            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights="distance",
                metric="euclidean"
            )

            cv_scores = cross_val_score(
                model,
                X_transformed,
                distances,
                cv=kf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            cv_mae = -cv_scores.mean()

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_params = {
                    "pca": pca_var,
                    "n_neighbors": n_neighbors
                }

    print(f"\n[BEST CV MAE]: {best_mae:.4f}")

    # -------------------------
    # Final model
    # -------------------------
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(images)

    final_pca = PCA(
        n_components=best_params["pca"],
        svd_solver="full",
        random_state=42
    )
    X_final = final_pca.fit_transform(X_scaled)

    final_model = KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights="distance",
        metric="euclidean"
    )

    final_model.fit(X_final, distances)

    # -------------------------
    # Test
    # -------------------------
    test_images = load_test_dataset(config)
    test_scaled = final_scaler.transform(test_images)
    test_pca = final_pca.transform(test_scaled)

    test_pred = final_model.predict(test_pca)

    save_results(test_pred)

    end_time = time.time()
    print(f"[INFO] Total runtime: {end_time - start_time:.2f} seconds")
