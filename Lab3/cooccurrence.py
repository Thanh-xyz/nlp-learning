import re
import numpy as np
from collections import Counter
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def tokenize(text):
    if isinstance(text, list):
        return [str(token).lower() for token in text]
    return re.findall(r"\b[a-zA-Z0-9']+\b", str(text).lower())


def build_vocabulary(corpus, min_freq=1):
    counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        counts.update(tokens)
    vocab = [token for token, freq in counts.items() if freq >= min_freq]
    return sorted(list(set(vocab)))


def build_cooccurrence_matrix(corpus, vocab, window_size=1, symmetric=True, return_sparse=False):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    vocab_set = set(vocab)
    vocab_size = len(vocab)

    cooccur_counts = Counter()

    for doc in corpus:
        tokens = tokenize(doc)
        indices = [vocab_index[t] for t in tokens if t in vocab_set]
        num_tokens = len(indices)

        for i in range(num_tokens):
            target_idx = indices[i]
            left_bound = max(0, i - window_size)
            right_bound = min(num_tokens, i + window_size + 1)

            for j in range(left_bound, right_bound):
                if i != j:
                    context_idx = indices[j]
                    cooccur_counts[(target_idx, context_idx)] += 1

    if return_sparse:
        rows = []
        cols = []
        data = []
        for (r, c), val in cooccur_counts.items():
            rows.append(r)
            cols.append(c)
            data.append(val)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(vocab_size, vocab_size),
            dtype=np.float64
        )
        return matrix

    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for (r, c), val in cooccur_counts.items():
        matrix[r, c] = val

    return matrix


def cosine_similarity(vec_a, vec_b):
    if sparse.issparse(vec_a):
        vec_a = vec_a.toarray().flatten()
    else:
        vec_a = np.asarray(vec_a, dtype=np.float64).flatten()

    if sparse.issparse(vec_b):
        vec_b = vec_b.toarray().flatten()
    else:
        vec_b = np.asarray(vec_b, dtype=np.float64).flatten()

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def get_word_vector(word, matrix, vocab):
    word = word.lower()
    if word not in vocab:
        raise ValueError(f"Word '{word}' not in vocabulary.")
    idx = vocab.index(word)
    if sparse.issparse(matrix):
        return matrix[idx].toarray().flatten()
    return np.asarray(matrix[idx], dtype=np.float64).flatten()


def word_similarity(word1, word2, matrix, vocab):
    word1 = word1.lower()
    word2 = word2.lower()
    if word1 not in vocab or word2 not in vocab:
        return 0.0
    vec1 = get_word_vector(word1, matrix, vocab)
    vec2 = get_word_vector(word2, matrix, vocab)
    return cosine_similarity(vec1, vec2)


def most_similar(word, matrix, vocab, top_k=5):
    word = word.lower()
    if word not in vocab:
        return []

    target_idx = vocab.index(word)
    target_vec = get_word_vector(word, matrix, vocab)
    norm_target = np.linalg.norm(target_vec)

    if norm_target == 0.0:
        return []

    if sparse.issparse(matrix):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = np.asarray(matrix, dtype=np.float64)

    dot_products = np.dot(dense_matrix, target_vec)
    norms = np.linalg.norm(dense_matrix, axis=1)
    norms[norms == 0.0] = 1e-9

    similarities = dot_products / (norms * norm_target)
    similarities[target_idx] = -1.0

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(vocab[idx], float(similarities[idx])) for idx in top_indices if similarities[idx] > -0.99]
    return results


def reduce_dimension_svd(matrix, n_components=50, random_state=42):
    n_features = matrix.shape[1]
    n_comp = min(n_components, n_features - 1) if n_features > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    reduced_embeddings = svd.fit_transform(matrix)
    return reduced_embeddings


def calculate_matrix_stats(matrix):
    if sparse.issparse(matrix):
        total_entries = matrix.shape[0] * matrix.shape[1]
        non_zeros = matrix.nnz
    else:
        total_entries = matrix.size
        non_zeros = int(np.count_nonzero(matrix))

    sparsity = 1.0 - (non_zeros / total_entries) if total_entries > 0 else 0.0
    return {
        "shape": matrix.shape,
        "total_entries": total_entries,
        "non_zeros": non_zeros,
        "sparsity": sparsity
    }


def run_unit_tests():
    toy_corpus = [
        "the cat eats fish",
        "the dog eats fish",
        "the cat likes milk",
        "the dog likes meat"
    ]
    vocab = ["cat", "dog", "eats", "likes", "fish", "milk", "meat"]
    matrix_w1 = build_cooccurrence_matrix(toy_corpus, vocab, window_size=1)

    v_cat = get_word_vector("cat", matrix_w1, vocab)
    v_dog = get_word_vector("dog", matrix_w1, vocab)
    sim_cat_dog = cosine_similarity(v_cat, v_dog)
    assert abs(sim_cat_dog - 1.0) < 1e-6

    sim_diff = word_similarity("eats", "likes", matrix_w1, vocab)
    assert 0.0 <= sim_diff <= 1.0

    dense_emb = reduce_dimension_svd(matrix_w1, n_components=3)
    assert dense_emb.shape == (len(vocab), 3)
    return True


if __name__ == "__main__":
    run_unit_tests()
    print("Unit tests passed successfully.")
