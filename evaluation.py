"""
Authors:
    - Lauren Lissette Llauradó Reyes
    - Carlos Alberto Sánchez Calderón
    - José Ángel Schiaffini Rodríguez
    - Karla Stefania Cruz Muñiz
Date:
    2025-05-15
Description:
    This script evaluates all models from models.py with both TF-IDF and BOW decoders.
    It tracks AUC scores to determine the best combination.
"""

import pandas as pd  # type:ignore
from prepro import prepro
from decoder import create_tfidf_features, create_bow_features
from models import Model, train_and_evaluate_model
from time import time
import matplotlib.pyplot as plt


def evaluation():
    """
    Function that evaluates all models with different feature extraction methods.
    Tests both TF-IDF and BOW vectorization, and evaluates all models
    defined in the Model enum.
    """
    # Load and preprocess training data
    print("Loading and preprocessing data...")
    start_time = time()
    df = pd.read_csv('./data_train(in).csv', encoding='latin-1')
    df = prepro(df)
    y = df['is_suicide'].to_numpy()
    print(f"Data loaded and preprocessed in {time() - start_time:.2f} seconds")

    # Setup for tracking results
    results = []

    # Define decoders and their names
    decoders = [
        ("TF-IDF", create_tfidf_features),
        ("BOW", create_bow_features)
    ]

    # # SVD parameters
    # n_components = 100

    # Evaluate all combinations
    for decoder_name, decoder_func in decoders:
        print(f"\n{'='*50}")
        print(f"EVALUATING WITH {decoder_name} VECTORIZATION")
        print(f"{'='*50}")

        # Create features
        print(f"\nCreating {decoder_name} features...")
        X, _ = decoder_func(df)
        print(f"Feature matrix shape: {X.shape}")

        # # Apply SVD
        # print(f"\nApplying SVD with {n_components} components...")
        # X_svd = randomized_svd_transformer(
        #     X=X, n_components=n_components, random_state=42)
        # print(f"Reduced feature matrix shape: {X_svd.shape}")

        # Train and evaluate each model
        for model in Model:
            print(f"\n{'='*30}")
            print(f"EVALUATING: {decoder_name} + {model.value}")
            print(f"{'='*30}")

            # Train model and capture AUC scores
            start_time = time()
            _, auc = train_and_evaluate_model(X, y, model)

            # Store results
            results.append({
                'decoder': decoder_name,
                'model': model.value,
                'auc': auc,
                'runtime': time() - start_time
            })

            print(f"\nRuntime: {results[-1]['runtime']:.2f} seconds")
            print(f"Full dataset AUC: {auc:.2f}")

    # Display summary of results
    print("\n\n")
    print(f"{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")

    # Create a DataFrame for easier sorting and display
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc', ascending=False)

    # Print all results
    print("\nAll combinations sorted by AUC score:")
    for i, row in results_df.iterrows():
        print(
            f"{row['decoder']} + {row['model']}: AUC = {row['auc']:.2f}, Runtime = {row['runtime']:.2f}s")

    # Get and print best combination
    best = results_df.iloc[0]
    print(f"\n{'='*50}")
    print(f"BEST COMBINATION: {best['decoder']} + {best['model']}")
    print(f"AUC Score: {best['auc']:.2f}")
    print(f"{'='*50}")

    # Plot results
    plt.figure(figsize=(12, 8))

    # Group by decoder and model
    pivoted = results_df.pivot(index='model', columns='decoder', values='auc')

    # Plot bar chart
    ax = pivoted.plot(kind='bar', width=0.7)
    plt.title('AUC Scores by Model and Decoder')
    plt.ylabel('AUC Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # AUC ranges from 0.5 (random) to 1.0 (perfect)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


if __name__ == "__main__":
    evaluation()
