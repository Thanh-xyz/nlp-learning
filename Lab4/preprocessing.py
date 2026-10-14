
import re
import html
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


def clean_text(text: str, remove_html: bool = True, lowercase: bool = True) -> str:

    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)
        
    text = re.sub(r"\s+", " ", text).strip()
    
    if lowercase:
        text = text.lower()
        
    return text


def tokenize(text: str, preserve_negations: bool = True) -> List[str]:

    cleaned = clean_text(text, lowercase=True)
    if preserve_negations:
        # Match words with optional single internal apostrophe (e.g., don't, can't)
        pattern = r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"
    else:
        pattern = r"\b[a-zA-Z]+\b"
        
    tokens = re.findall(pattern, cleaned)
    return tokens


def compute_average_word2vec(
    tokens: List[str],
    w2v_model: Word2Vec,
    vector_size: int = 100
) -> np.ndarray:

    wv = w2v_model.wv
    vectors = [wv[w] for w in tokens if w in wv]
    if len(vectors) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def get_average_w2v_matrix(
    texts: List[str],
    w2v_model: Word2Vec,
    vector_size: int = 100
) -> np.ndarray:

    matrix = np.zeros((len(texts), vector_size), dtype=np.float32)
    for i, doc in enumerate(texts):
        tokens = tokenize(doc)
        matrix[i] = compute_average_word2vec(tokens, w2v_model, vector_size=vector_size)
    return matrix


def split_dataset(
    df: pd.DataFrame,
    text_col: str = "review",
    label_col: str = "sentiment",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:

    X_temp, X_test, y_temp, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]
    )
    
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    sample_text = "This film is <b>NOT</b> good! It's an absolute waste of time... <br /><br />Avoid it."
    print("Original text:")
    print(sample_text)
    
    cleaned = clean_text(sample_text)
    print("\nCleaned text:")
    print(cleaned)
    
    tokens = tokenize(sample_text)
    print("\nTokens:")
    print(tokens)
