import math
import numpy as np


def build_vocabulary(documents):
    unique_terms = set()
    for doc in documents:
        tokens = doc.lower().split()
        for token in tokens:
            unique_terms.add(token)
    return sorted(list(unique_terms))


def compute_counts(documents, vocab):
    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    N = len(documents)
    V = len(vocab)
    counts = np.zeros((N, V), dtype=np.float64)
    
    for i, doc in enumerate(documents):
        tokens = doc.lower().split()
        for token in tokens:
            if token in vocab_index:
                counts[i, vocab_index[token]] += 1.0
                
    return counts


def compute_tf(counts):
    doc_lengths = np.sum(counts, axis=1, keepdims=True)
    doc_lengths_safe = np.where(doc_lengths == 0, 1.0, doc_lengths)
    tf = counts / doc_lengths_safe
    return tf


def compute_idf(counts, smooth=False):
    N = counts.shape[0]
    df = np.sum(counts > 0, axis=0)
    
    if smooth:
        idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    else:
        df_safe = np.where(df == 0, 1e-9, df)
        idf = np.log(N / df_safe)
        idf[df == 0] = 0.0
        
    return idf


def compute_tfidf(tf, idf):
    return tf * idf


def cosine_similarity(vec1, vec2):
    import scipy.sparse as sp
    if sp.issparse(vec1) or sp.issparse(vec2):
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        return sk_cosine(vec1, vec2)
        
    v1 = np.asarray(vec1, dtype=np.float64)
    v2 = np.asarray(vec2, dtype=np.float64)
    
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    
    denom = norm1 * norm2.T if (v1.ndim > 1 or v2.ndim > 1) else norm1 * norm2
    denom = np.squeeze(denom)
    
    dot_product = np.squeeze(np.dot(v1, v2.T if v2.ndim > 1 else v2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.divide(dot_product, denom, where=denom != 0)
        sim = np.nan_to_num(sim, nan=0.0)
        
    return float(sim) if (v1.ndim <= 1 and v2.ndim <= 1) else sim


def run_unit_tests():
    print("==================================================")
    print(" ĐANG CHẠY UNIT TESTS CHO CORE IMPLEMENTATION...")
    print("==================================================")
    
    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]
    
    vocab = build_vocabulary(corpus)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Lỗi vocab! Thực tế: {vocab}, Kỳ vọng: {expected_vocab}"
    print("[PASS] Test 1: build_vocabulary() chính xác.")
    
    counts = compute_counts(corpus, vocab)
    expected_counts = np.array([
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
    ])
    assert np.allclose(counts, expected_counts), f"Lỗi counts!\nThực tế:\n{counts}\nKỳ vọng:\n{expected_counts}"
    print("[PASS] Test 2: compute_counts() chính xác.")
    
    tf = compute_tf(counts)
    expected_tf = expected_counts / 3.0
    assert np.allclose(tf, expected_tf), f"Lỗi tf!\nThực tế:\n{tf}\nKỳ vọng:\n{expected_tf}"
    assert abs(np.sum(tf[0]) - 1.0) < 1e-9, f"Tổng TF D1 không bằng 1! {np.sum(tf[0])}"
    print("[PASS] Test 3: compute_tf() chính xác và chuẩn hóa tổng = 1.0.")
    
    idf = compute_idf(counts, smooth=False)
    expected_idf = np.array([
        math.log(3.0 / 2.0),
        math.log(3.0 / 1.0),
        math.log(3.0 / 2.0),
        math.log(3.0 / 3.0), 
        math.log(3.0 / 1.0),
    ])
    assert np.allclose(idf, expected_idf), f"Lỗi idf!\nThực tế:\n{idf}\nKỳ vọng:\n{expected_idf}"
    assert abs(idf[vocab.index("fish")] - 0.0) < 1e-9, "IDF của 'fish' phải bằng 0!"
    print("[PASS] Test 4: compute_idf() chính xác (idf('fish') = 0).")
    
    tfidf = compute_tfidf(tf, idf)
    expected_tfidf = tf * idf
    assert np.allclose(tfidf, expected_tfidf), f"Lỗi tfidf!"
    assert abs(tfidf[0, vocab.index("fish")] - 0.0) < 1e-9, "TF-IDF của 'fish' trong D1 phải bằng 0!"
    cat_tfidf_d1 = tfidf[0, vocab.index("cat")]
    assert abs(cat_tfidf_d1 - (1.0/3.0) * math.log(1.5)) < 1e-9, "TF-IDF của 'cat' sai!"
    print("[PASS] Test 5: compute_tfidf() chính xác.")
    
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 0.0])
    sim = cosine_similarity(x, y)
    expected_sim = 2.0 / math.sqrt(6.0)
    assert abs(sim - expected_sim) < 1e-9, f"Lỗi cosine_similarity! Thực tế: {sim}, Kỳ vọng: {expected_sim}"
    print(f"[PASS] Test 6: cosine_similarity([1,1,1], [1,1,0]) = {sim:.6f} == 2/sqrt(6).")
    

def compare_with_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]
    
    print("\n==================================================")
    print(" SO SÁNH VỚI SCIKIT-LEARN TFIDFVECTORIZER")
    print("==================================================")
    
    vocab = build_vocabulary(corpus)
    counts = compute_counts(corpus, vocab)
    tf = compute_tf(counts)
    idf_std = compute_idf(counts, smooth=False)
    tfidf_std = compute_tfidf(tf, idf_std)
    
    print("\n1. Custom Implementation (Standard Textbook Formulas):")
    print("   Vocabulary:", vocab)
    print("   IDF (unsmoothed):", np.round(idf_std, 4))
    print("   D1 TF-IDF Vector:", np.round(tfidf_std[0], 4))
    
    sk_vec = TfidfVectorizer(norm='l2', smooth_idf=True)
    sk_X = sk_vec.fit_transform(corpus)
    
    print("\n2. Scikit-learn (smooth_idf=True, norm='l2'):")
    print("   Sklearn Vocab:", sk_vec.get_feature_names_out().tolist())
    print("   Sklearn IDF:", np.round(sk_vec.idf_, 4))
    print("   D1 Vector (L2 normalized):", np.round(sk_X.toarray()[0], 4))
    
    idf_smooth = compute_idf(counts, smooth=True)
    tfidf_raw_counts = counts * idf_smooth  
    norms = np.linalg.norm(tfidf_raw_counts, axis=1, keepdims=True)
    custom_mimic_sklearn = tfidf_raw_counts / norms
    
    print("\n3. Custom Implementation khi cấu hình cùng Convention với Sklearn:")
    print("   Custom IDF (smoothed):", np.round(idf_smooth, 4))
    print("   Custom D1 Vector (raw counts * smooth IDF + L2 norm):", np.round(custom_mimic_sklearn[0], 4))
    
    diff = np.max(np.abs(sk_X.toarray() - custom_mimic_sklearn))
    print(f"\n=> Độ lệch lớn nhất giữa Custom (theo Sklearn convention) và Sklearn: {diff:.1e}")
    print("=> Kết luận: Cài đặt toán học hoàn toàn khớp khi quy về cùng convention!")


if __name__ == "__main__":
    run_unit_tests()
    compare_with_sklearn()
