from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb
from enum import Enum

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


class Model(Enum):
    KNN = "KNN"  # Changed from KMEANS to KNN
    NN = "NeuralNetwork"


def train_and_evaluate_model(X, y, model: Model):
    """
    Trains and evaluates either KNN or a Neural Network with hyperparameter tuning.

    Args:
        X: Feature matrix
        y: Target labels
        model: Model type (KNN or NN)

    Returns:
        best_model: The best trained model
        best_score: The best performance metric
    """

    match model:
        case Model.KNN:
            return train_knn(X, y)  # Changed from train_kmeans to train_knn
        case Model.NN:
            return train_neural_network(X, y)


def train_knn(X, y):
    """Train KNN classifier with different numbers of neighbors"""
    print("\n=== Training KNN with different neighbor counts ===")

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        X.toarray() if hasattr(X, "toarray") else X)

    # Prepare for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Try different neighbor counts
    k_range = [3, 5, 7, 9, 11, 15, 21]
    results = []

    for n_neighbors in k_range:
        # Setup KNN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Cross validation metrics
        cv_auc = []
        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train and predict
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            # Probability for positive class
            y_proba = knn.predict_proba(X_val)[:, 1]

            # Calculate metrics
            cv_auc.append(roc_auc_score(y_val, y_proba))
            cv_accuracy.append(accuracy_score(y_val, y_pred))
            cv_precision.append(precision_score(y_val, y_pred))
            cv_recall.append(recall_score(y_val, y_pred))
            cv_f1.append(f1_score(y_val, y_pred))

        # Calculate mean metrics
        mean_auc = np.mean(cv_auc)
        mean_accuracy = np.mean(cv_accuracy)
        mean_precision = np.mean(cv_precision)
        mean_recall = np.mean(cv_recall)
        mean_f1 = np.mean(cv_f1)

        # Add to results
        results.append({
            'n_neighbors': n_neighbors,
            'auc': mean_auc,
            'accuracy': mean_accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': mean_f1
        })

        print(
            f"k={n_neighbors}: AUC={mean_auc:.3f}, Accuracy={mean_accuracy:.3f}, F1={mean_f1:.3f}")

    # Visualize results
    plt.figure(figsize=(15, 4))

    # AUC (higher is better)
    plt.plot([r['n_neighbors'] for r in results], [r['auc']
             for r in results], 'o-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('AUC Score')
    plt.title('AUC vs Neighbor Count')

    plt.tight_layout()
    plt.savefig('knn_evaluation.png', dpi=300)
    plt.show()

    # Find best model based on AUC (higher is better)
    best_result = max(results, key=lambda x: x['auc'])
    best_k = best_result['n_neighbors']
    best_auc = best_result['auc']

    print(f"Best KNN model by AUC: k={best_k} (AUC: {best_auc:.3f})")

    # Train final model with best k on full dataset
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_scaled, y)

    # Evaluate on training data (just for reporting)
    y_pred = best_model.predict(X_scaled)
    y_proba = best_model.predict_proba(X_scaled)[:, 1]

    print("\n=== KNN final metrics on training data ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"Precision: {precision_score(y, y_pred):.2f}")
    print(f"Recall: {recall_score(y, y_pred):.2f}")
    print(f"F1-score: {f1_score(y, y_pred):.2f}")
    print(f"AUC: {roc_auc_score(y, y_proba):.2f}")

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
            input_layer = Input(shape=(X_train.shape[1],))
            x = Dense(64, activation=activation)(input_layer)
            x = Dropout(0.3)(x)
            x = Dense(32, activation=activation)(x)
            x = Dropout(0.2)(x)
            output = Dense(1, activation='sigmoid')(x)
            model = tf.keras.models.Model(inputs=input_layer, outputs=output)

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
    plt.figure(figsize=(10, 5))

    # Plot only AUC comparison
    x = np.arange(len(activations))

    plt.bar(
        x,
        [r['metrics']['auc'] for r in results],
        width=0.4,
        color='blue'
    )

    plt.xlabel('Activation Function')
    plt.ylabel('AUC Score')
    plt.title('Neural Network AUC Performance by Activation Function')
    plt.xticks(x, activations)
    plt.ylim(0.5, 1.0)  # AUC range from 0.5 to 1.0
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
    input_layer = Input(shape=(X_scaled.shape[1],))
    x = Dense(64, activation=best_activation)(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(32, activation=best_activation)(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    final_model = tf.keras.models.Model(inputs=input_layer, outputs=output)

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
    Evaluates either a KNN or Neural Network model on test data

    Args:
        clf: The trained model
        X_test: Test features
        y_test: True labels
        model_type: Type of model (KNN or NN)
    """
    # Convert and scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test)

    if model_type == Model.KNN:
        # For KNN, predict both classes and probabilities
        y_pred = clf.predict(X_scaled)
        # Probability for positive class
        y_proba = clf.predict_proba(X_scaled)[:, 1]

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

    # Create ROC curve
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
    print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")
