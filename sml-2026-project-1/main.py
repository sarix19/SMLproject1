from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

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

    # ─────────────────────────────────────────
    # 2. PREPROCESSING: StandardScaler
    #    fit() NUR auf Trainingsdaten!
    # ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # ─────────────────────────────────────────
    # 3. DIMENSIONSREDUKTION: PCA
    #    95% der Varianz behalten
    #    fit() NUR auf Trainingsdaten!
    # ─────────────────────────────────────────
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    print(f"[INFO]: PCA: {X_train.shape[1]} Features → {X_train_pca.shape[1]} Komponenten")

    # ─────────────────────────────────────────
    # 4. MODELL + HYPERPARAMETER-TUNING
    #    GridSearchCV mit cv=3 auf Trainingsdaten
    # ─────────────────────────────────────────
    model = GradientBoostingRegressor(random_state=42)

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("[INFO]: Starte GridSearch...")
    grid_search.fit(X_train_pca, y_train)

    best_model = grid_search.best_estimator_
    print(f"[INFO]: Beste Parameter: {grid_search.best_params_}")

    # ─────────────────────────────────────────
    # 5. EVALUATION auf Validierungsset
    # ─────────────────────────────────────────
    val_pred  = best_model.predict(X_val_pca)
    train_pred = best_model.predict(X_train_pca)

    print("\n[RESULTS] Training:")
    print_results(y_train, train_pred)
    print("[RESULTS] Validation:")
    print_results(y_val, val_pred)

    # ─────────────────────────────────────────
    # 6. TEST-DATEN LADEN & VORHERSAGEN
    # ─────────────────────────────────────────
    test_images = load_test_dataset(config)
    test_images = np.array(test_images)

    test_scaled = scaler.transform(test_images)
    test_pca    = pca.transform(test_scaled)
    test_pred   = best_model.predict(test_pca)

    # Save the results
    save_results(test_pred)
    print("[INFO]: Vorhersagen gespeichert in prediction.csv")