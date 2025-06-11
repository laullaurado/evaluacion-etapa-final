"""
Authors:
    - José Ángel Schiaffini Rodríguez
    - Karla Stefania Cruz Muñiz
Date:
    2025-05-14
Description:
    Suicide ideation detection system using machine learning techniques.
    This script implements a pipeline that processes text data, extracts features,
    applies dimensionality reduction with SVD, trains a classifier, and evaluates
    model performance on test data to identify potential suicide ideation in text content.
"""

import pandas as pd  # type: ignore
from prepro import prepro
from decoder import create_tfidf_features, create_bow_features
from models import Model, train_and_evaluate_model, evaluate_model


def main():
    """
    Main function for suicide ideation detection pipeline.
    Loads training and test data, preprocesses text, extracts features,
    trains a model, and evaluates performance on test data.
    """
    
    # Training data
    df_train = pd.read_csv('./data_train(in).csv', encoding='latin-1')
    df_train = prepro(df_train)

    X_train, vectorizer = create_tfidf_features(df_train)

    # X_train = randomized_svd_transformer(X=X_train, random_state=42)

    clf, _ = train_and_evaluate_model(X_train, df_train['is_suicide'], Model.LR)

    # Test data
    df_test = pd.read_csv('./data_test_fold1(in).csv', encoding='latin-1')
    df_test = prepro(df_test)

    X_test = vectorizer.transform(df_test['text_clean'])

    # X_test = randomized_svd_transformer(X=X_test, random_state=42)

    evaluate_model(clf, X_test, df_test['is_suicide'])


if __name__ == "__main__":
    main()
