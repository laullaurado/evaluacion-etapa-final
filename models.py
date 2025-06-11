"""
Authors:
    - Lauren Lissette Llauradó Reyes
    - Carlos Alberto Sánchez Calderón
Date:
    2025-05-14
Description:
    This script defines a machine learning pipeline for suicide ideation detection.
    It includes a function to train and evaluate models.
"""

# from sklearn.model_selection import StratifiedKFold  # type: ignore
# from sklearn.tree import DecisionTreeClassifier  # type: ignore
# from sklearn.linear_model import LogisticRegression  # type: ignore
# from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC  # type: ignore
import xgboost as xgb  # type: ignore
from enum import Enum

# Add these to your existing imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class Model(Enum):
    KMEANS = "KMeans"
    NN = "NeuralNetwork"


def train_and_evaluate_model(X_tfidf, y, model: Model):
    """
    Trains and evaluates either K-means clustering or a Neural Network with hyperparameter tuning.

    Args:
        X_tfidf: Feature matrix
        y: Target labels
        model: Model type (KMEANS or NN)

    Returns:
        best_model: The best trained model
        best_score: The best performance metric
    """

    match model:
        case Model.KMEANS:
            return train_kmeans(X_tfidf, y)
        case Model.NN:
            return train_neural_network(X_tfidf, y)


def train_kmeans(X, y):
    """Train K-means clustering with different numbers of clusters"""
    print("\n=== Training K-means with different cluster counts ===")

    # Standardize the features (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.toarray())

    # Try different cluster counts
    cluster_range = [2, 3, 4, 5, 8, 10]
    results = []

    for n_clusters in cluster_range:
        # Train K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Evaluate using silhouette score and Davies-Bouldin index
        silhouette = silhouette_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

        # Calculate classification metrics by mapping clusters to classes
        cluster_to_label = {}
        for cluster in range(n_clusters):
            mask = (cluster_labels == cluster)
            if mask.sum() > 0:
                cluster_to_label[cluster] = np.argmax(np.bincount(y[mask]))

        # Map predictions to labels
        y_pred = np.array([cluster_to_label.get(label, 0)
                          for label in cluster_labels])

        # Create probability estimate for AUC calculation
        y_proba = np.zeros((len(y), 2))
        y_proba[np.arange(len(y)), y_pred] = 1
        y_proba_positive = y_proba[:, 1]  # Probability of positive class

        # Calculate AUC if possible (requires both classes to be present in predictions)
        auc = 0.5  # Default value (random classifier)
        if len(np.unique(y_pred)) > 1:
            auc = roc_auc_score(y, y_proba_positive)

        # Add to results
        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'inertia': kmeans.inertia_,
            'model': kmeans,
            'auc': auc,
            'y_pred': y_pred,
            'cluster_to_label': cluster_to_label
        })

        print(
            f"Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}, AUC: {auc:.3f}")

    # Visualize results
    plt.figure(figsize=(15, 4))

    # Silhouette scores (higher is better)
    plt.subplot(1, 4, 1)
    plt.plot([r['n_clusters'] for r in results], [r['silhouette']
             for r in results], 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Cluster Count')

    # Davies-Bouldin index (lower is better)
    plt.subplot(1, 4, 2)
    plt.plot([r['n_clusters'] for r in results], [r['davies_bouldin']
             for r in results], 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index vs Cluster Count')

    # Inertia (elbow method)
    plt.subplot(1, 4, 3)
    plt.plot([r['n_clusters'] for r in results], [r['inertia']
             for r in results], 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # AUC (higher is better)
    plt.subplot(1, 4, 4)
    plt.plot([r['n_clusters'] for r in results], [r['auc']
             for r in results], 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('AUC Score')
    plt.title('AUC vs Cluster Count')

    plt.tight_layout()
    plt.savefig('kmeans_evaluation.png', dpi=300)
    plt.show()

    # Find best model based on silhouette score (higher is better)
    best_result_silhouette = max(results, key=lambda x: x['silhouette'])

    # Find best model based on AUC (higher is better)
    best_result_auc = max(results, key=lambda x: x['auc'])

    best_model = best_result_auc['model']
    best_auc = best_result_auc['auc']

    print(
        f"\nBest K-means model by silhouette: {best_result_silhouette['n_clusters']} clusters (score: {best_result_silhouette['silhouette']:.3f})")
    print(
        f"Best K-means model by AUC: {best_result_auc['n_clusters']} clusters (AUC: {best_auc:.3f})")

    # Use the best model by AUC
    cluster_to_label = best_result_auc['cluster_to_label']
    y_pred = best_result_auc['y_pred']

    print("\n=== K-means as classifier ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"Precision: {precision_score(y, y_pred):.2f}")
    print(f"Recall: {recall_score(y, y_pred):.2f}")
    print(f"F1-score: {f1_score(y, y_pred):.2f}")
    print(f"AUC: {best_auc:.2f}")

    return best_model, best_auc


def train_neural_network(X, y):
    """Train a 3-layer Neural Network with different activation functions"""
    print("\n=== Training Neural Network with different activation functions ===")

    # Convert sparse matrix to dense if needed and standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        X.toarray() if hasattr(X, "toarray") else X)

    # Prepare for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=80)

    # Hyperparameters to test
    activations = ['relu', 'tanh', 'sigmoid']
    results = []

    for activation in activations:
        print(f"\n--- Testing activation: {activation} ---")

        all_metrics = {
            'auc': [], 'accuracy': [], 'precision': [],
            'recall': [], 'f1': [], 'val_loss': []
        }

        all_cm = np.zeros((2, 2), int)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_val = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]

            # Build model
            model = Sequential([
                Dense(64, activation=activation,
                      input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(32, activation=activation),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Train with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )

            # Evaluate
            y_proba = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)

            cm = confusion_matrix(y_val, y_pred)
            all_cm += cm

            all_metrics['auc'].append(roc_auc_score(y_val, y_proba))
            all_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            all_metrics['precision'].append(precision_score(y_val, y_pred))
            all_metrics['recall'].append(recall_score(y_val, y_pred))
            all_metrics['f1'].append(f1_score(y_val, y_pred))
            all_metrics['val_loss'].append(min(history.history['val_loss']))

        # Calculate mean metrics
        mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        print(f"Mean AUC: {mean_metrics['auc']:.3f}")
        print(f"Mean Accuracy: {mean_metrics['accuracy']:.3f}")
        print(f"Mean F1-score: {mean_metrics['f1']:.3f}")

        results.append({
            'activation': activation,
            'metrics': mean_metrics,
            'cm': all_cm
        })

    # Visualize results
    plt.figure(figsize=(12, 5))

    # Plot metrics comparison
    metrics_to_plot = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(activations))
    width = 0.15

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(
            x + (i - len(metrics_to_plot)/2 + 0.5) * width,
            [r['metrics'][metric] for r in results],
            width=width,
            label=metric.capitalize()
        )

    plt.xlabel('Activation Function')
    plt.ylabel('Score')
    plt.title('Neural Network Performance by Activation Function')
    plt.xticks(x, activations)
    plt.legend()
    plt.tight_layout()
    plt.savefig('nn_activation_comparison.png', dpi=300)
    plt.show()

    # Find best model based on AUC
    best_result = max(results, key=lambda x: x['metrics']['auc'])
    best_activation = best_result['activation']
    best_auc = best_result['metrics']['auc']

    print(f"\nBest activation function: {best_activation}")
    print(f"Best AUC score: {best_auc:.3f}")

    # Build final model with best activation
    final_model = Sequential([
        Dense(64, activation=best_activation,
              input_shape=(X_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation=best_activation),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    final_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train on full dataset
    final_model.fit(
        X_scaled, y,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    # Display confusion matrix for best activation
    disp = ConfusionMatrixDisplay(
        confusion_matrix=best_result['cm'],
        display_labels=['no', 'yes']
    )
    disp.plot(cmap=plt.colormaps["Blues"])
    plt.title(f"Confusion Matrix - {best_activation} activation")
    plt.savefig('nn_confusion_matrix.png')
    plt.show()

    return final_model, best_auc


def evaluate_model(clf, X_test, y_test, model_type: Model):
    """
    Evaluates either a K-means or Neural Network model on test data

    Args:
        clf: The trained model
        X_test: Test features
        y_test: True labels
        model_type: Type of model (KMEANS or NN)
    """
    # Convert and scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test)

    if model_type == Model.KMEANS:
        # For K-means, map clusters to labels first
        cluster_labels = clf.predict(X_scaled)

        # Map clusters to majority class labels
        cluster_to_label = {}
        for cluster in range(clf.n_clusters):
            # Use training labels from the fitted model
            mask = (clf.labels_ == cluster)
            if mask.sum() > 0:
                cluster_to_label[cluster] = np.argmax(
                    np.bincount(y_test[mask]))

        y_pred = np.array([cluster_to_label.get(label, 0)
                          for label in cluster_labels])
        y_proba = np.zeros((len(y_test), 2))
        y_proba[np.arange(len(y_test)), y_pred] = 1
        y_proba = y_proba[:, 1]  # Get probabilities for positive class

    elif model_type == Model.NN:
        # For Neural Network, get predictions directly
        y_proba = clf.predict(X_scaled, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.colormaps["Blues"])
    plt.title("Test Confusion Matrix")
    plt.savefig('confusion_matrix_test.png')
    plt.show()

    # Create ROC curve if applicable
    if model_type == Model.NN or (model_type == Model.KMEANS and len(np.unique(y_pred)) > 1):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray',
                 linestyle='--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig('roc_curve_test.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Print metrics
    print("\n=== Test Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2f}")

    if model_type == Model.NN or (model_type == Model.KMEANS and len(np.unique(y_pred)) > 1):
        print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")
