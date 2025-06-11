"""
Authors:
    - Carlos Alberto Sánchez Calderón
    - Karla Stefania Cruz Muñiz
Date:
    2025-05-14
Description:
    This module contains a function to preprocess text data
"""

import re
import pandas as pd  # type: ignore
import ftfy  # type: ignore
import contractions  # type: ignore
import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import spacy  # type: ignore

# Download required NLTK data
nltk.download('stopwords')

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')


def clean_text(text: str) -> str:
    """
    Cleans and preprocesses a given text string.
    This function performs the following operations on the input text:
    1. Fixes mojibake and broken encodings using the `ftfy` library.
    2. Expands contractions (e.g., "don't" -> "do not") using the `contractions` library.
    3. Removes special characters and symbols, retaining only alphabetic characters and spaces.
    4. Converts the text to lowercase.
    5. Replaces multiple spaces with a single space.
    6. Trims leading and trailing spaces.
    7. Removes English stopwords from the text.
    Args:
        text (str): The input text string to be cleaned.
    Returns:
        str: The cleaned and preprocessed text string.
    """
    stop_words = set(stopwords.words('english'))

    # Fix mojibake and broken encodings
    text = ftfy.fix_text(text)
    # Expand contractions
    text = contractions.fix(text)
    # Remove special characters and symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing spaces
    text = text.strip()
    # Remove English stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the input text by reducing each word to its base or dictionary form.
    Args:
        text (str): The input text to be lemmatized.
    Returns:
        str: A string containing the lemmatized version of the input text, 
                with words separated by spaces.
    """

    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def tokenize_text(text: str) -> list[str]:
    """
    Splits the given text string into individual words
    using whitespace as the delimiter.
    Args:
        text (str): The input string to be tokenized.
    Returns:
        list[str]: A list of words obtained by splitting the input text.
    """

    return text.split()


def transform_label(label: str) -> int:
    """
    Converts a textual label into a numerical representation.
    Args:
        label (str): The input label, expected to be either 'yes' or 'no'.
    Returns:
        int: Returns 1 if the label is 'yes', otherwise returns 0.
    """

    return 1 if label == 'yes' else 0


def prepro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame containing text data for suicide ideation detection.
    This function performs the following steps:
    1. Combines the 'title' and 'text' columns into a new column 'full_text'.
    2. Cleans the text data by:
        - Fixing mojibake and broken encodings.
        - Expanding contractions (e.g., "you'll" → "you will").
        - Removing special characters and symbols, keeping only letters and spaces.
        - Converting text to lowercase.
        - Removing multiple spaces and trimming leading/trailing spaces.
        - Removing English stopwords.
    3. Lemmatizes the cleaned text to reduce words to their base forms.
    4. Tokenizes the cleaned text into a list of words.
    5. Transforms the 'is_suicide' column into a boolean label (True for 'yes', False otherwise).
    Args:
        df (pd.DataFrame): Input DataFrame containing the text data. 
                           Expected columns: 'title', 'text', and 'is_suicide'.
    Returns:
        pd.DataFrame: The processed DataFrame with the following additional columns:
                      - 'full_text': Combined text from 'title' and 'text'.
                      - 'text_clean': Cleaned and lemmatized text.
                      - 'is_suicide': Boolean label indicating suicide ideation.
                      - 'tokens': Tokenized list of words from the cleaned text.
    """

    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['text_clean'] = df['full_text'].astype(str).apply(clean_text)
    df['text_clean'] = df['text_clean'].astype(str).apply(lemmatize_text)
    df['is_suicide'] = df['is_suicide'].astype(str).apply(transform_label)
    df['tokens'] = df['text_clean'].astype(str).apply(tokenize_text)
    df[['full_text', 'text_clean', 'tokens']].head(10)
    return df
