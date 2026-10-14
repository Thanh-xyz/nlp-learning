import os
import sys
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from gensim.models import Word2Vec

# Set random seed
SEED = 42
np.random.seed(SEED)

print("--- 1. Loading Dataset ---")
df = pd.read_csv('dataset/IMDB Dataset.csv')
print(f"Dataset shape: {df.shape}")

# Deduplication note:
num_duplicates = df.duplicated(subset=['review']).sum()
print(f"Duplicate reviews: {num_duplicates}")

# Basic cleaning for tokenization
def clean_text_simple(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("--- 2. Dataset Inspection & Statistics ---")
lengths_words = df['review'].apply(lambda x: len(x.split()))
print(f"Class distribution:\n{df['sentiment'].value_counts()}")
print(f"Percentage:\n{df['sentiment'].value_counts(normalize=True)*100}")
print(f"Word length describe:\n{lengths_words.describe()}")

# Plotting class distribution
os.makedirs('lab04', exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['#3b82f6', '#ef4444'], ax=ax, edgecolor='black', alpha=0.85)
ax.set_title("Class Distribution in IMDb Dataset (Balanced: 50% / 50%)", fontsize=12, fontweight='bold')
ax.set_xlabel("Sentiment", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_xticklabels(["Positive", "Negative"], rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,} ({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('lab04/class_distribution.png', dpi=300)
plt.close()

# Plotting document length distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lengths_words, bins=50, color='#6366f1', edgecolor='black', alpha=0.8)
ax.set_title("Document Length Distribution (Word Count)", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Words per Review", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.axvline(lengths_words.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f"Mean: {lengths_words.mean():.1f}")
ax.axvline(lengths_words.median(), color='gold', linestyle='-', linewidth=1.5, label=f"Median: {lengths_words.median():.1f}")
ax.legend()
plt.tight_layout()
plt.savefig('lab04/document_length_distribution.png', dpi=300)
plt.close()

print("--- 3. Proper Split (Train / Val / Test) To Avoid Data Leakage ---")
# Split: 70% Train, 10% Val, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=SEED, stratify=df['sentiment']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp
)
# Result: 35,000 train (70%), 5,000 val (10%), 10,000 test (20%)
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

results_list = []

print("--- 4. Experiment 2: Baseline (TF-IDF Unigram + Logistic Regression) ---")
t0 = time.time()
tfidf_unigram = TfidfVectorizer(max_features=25000, preprocessor=clean_text_simple, stop_words='english')
X_train_tfidf_uni = tfidf_unigram.fit_transform(X_train)
X_val_tfidf_uni = tfidf_unigram.transform(X_val)
X_test_tfidf_uni = tfidf_unigram.transform(X_test)
tfidf_fit_time = time.time() - t0

t1 = time.time()
clf_lr_baseline = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
clf_lr_baseline.fit(X_train_tfidf_uni, y_train)
lr_baseline_train_time = time.time() - t1
total_baseline_time = tfidf_fit_time + lr_baseline_train_time

y_pred_baseline = clf_lr_baseline.predict(X_test_tfidf_uni)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
prec_baseline = precision_score(y_test, y_pred_baseline, pos_label='positive')
rec_baseline = recall_score(y_test, y_pred_baseline, pos_label='positive')
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label='positive')

print(f"Baseline TF-IDF + LR -> Acc: {acc_baseline:.4f}, Prec: {prec_baseline:.4f}, Rec: {rec_baseline:.4f}, F1: {f1_baseline:.4f}, Time: {total_baseline_time:.2f}s")
results_list.append({
    "Model": "Baseline (TF-IDF Unigram + LR)",
    "Representation": "TF-IDF (Unigram)",
    "Classifier": "Logistic Regression",
    "Vocabulary_Size": len(tfidf_unigram.vocabulary_),
    "Accuracy": round(acc_baseline, 4),
    "Precision": round(prec_baseline, 4),
    "Recall": round(rec_baseline, 4),
    "F1_Score": round(f1_baseline, 4),
    "Training_Time_Sec": round(total_baseline_time, 2)
})

print("--- 5. Experiment 3: Model B (Average Word2Vec + Logistic Regression) ---")
# Tokenize train sentences for Word2Vec training
train_tokens = [re.findall(r"\b[a-zA-Z]+\b", clean_text_simple(s).lower()) for s in X_train]
t0 = time.time()
w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2, workers=4, epochs=5, seed=SEED)
w2v_train_time = time.time() - t0
print(f"Word2Vec trained in {w2v_train_time:.2f}s, vocab size: {len(w2v_model.wv)}")

def docs_to_avg_w2v(docs, w2v, dim=100):
    wv = w2v.wv
    matrix = np.zeros((len(docs), dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        tokens = re.findall(r"\b[a-zA-Z]+\b", clean_text_simple(doc).lower())
        vecs = [wv[w] for w in tokens if w in wv]
        if len(vecs) > 0:
            matrix[i] = np.mean(vecs, axis=0)
    return matrix

t0 = time.time()
X_train_w2v = docs_to_avg_w2v(X_train, w2v_model)
X_test_w2v = docs_to_avg_w2v(X_test, w2v_model)
feat_w2v_time = time.time() - t0

t1 = time.time()
clf_lr_w2v = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
clf_lr_w2v.fit(X_train_w2v, y_train)
lr_w2v_train_time = time.time() - t1
total_w2v_time = w2v_train_time + feat_w2v_time + lr_w2v_train_time

y_pred_w2v = clf_lr_w2v.predict(X_test_w2v)
acc_w2v = accuracy_score(y_test, y_pred_w2v)
prec_w2v = precision_score(y_test, y_pred_w2v, pos_label='positive')
rec_w2v = recall_score(y_test, y_pred_w2v, pos_label='positive')
f1_w2v = f1_score(y_test, y_pred_w2v, pos_label='positive')

print(f"Avg Word2Vec + LR -> Acc: {acc_w2v:.4f}, Prec: {prec_w2v:.4f}, Rec: {rec_w2v:.4f}, F1: {f1_w2v:.4f}, Time: {total_w2v_time:.2f}s")
results_list.append({
    "Model": "Model B (Avg Word2Vec + LR)",
    "Representation": "Average Word2Vec (100d)",
    "Classifier": "Logistic Regression",
    "Vocabulary_Size": len(w2v_model.wv),
    "Accuracy": round(acc_w2v, 4),
    "Precision": round(prec_w2v, 4),
    "Recall": round(rec_w2v, 4),
    "F1_Score": round(f1_w2v, 4),
    "Training_Time_Sec": round(total_w2v_time, 2)
})

print("--- 6. Experiment 3: Model C (TF-IDF Unigram + Linear SVM) ---")
t0 = time.time()
clf_svm_uni = LinearSVC(random_state=SEED, max_iter=2000, C=1.0)
clf_svm_uni.fit(X_train_tfidf_uni, y_train)
svm_uni_train_time = time.time() - t0 + tfidf_fit_time

y_pred_svm_uni = clf_svm_uni.predict(X_test_tfidf_uni)
acc_svm_uni = accuracy_score(y_test, y_pred_svm_uni)
prec_svm_uni = precision_score(y_test, y_pred_svm_uni, pos_label='positive')
rec_svm_uni = recall_score(y_test, y_pred_svm_uni, pos_label='positive')
f1_svm_uni = f1_score(y_test, y_pred_svm_uni, pos_label='positive')

print(f"TF-IDF Unigram + Linear SVM -> Acc: {acc_svm_uni:.4f}, Prec: {prec_svm_uni:.4f}, Rec: {rec_svm_uni:.4f}, F1: {f1_svm_uni:.4f}, Time: {svm_uni_train_time:.2f}s")
results_list.append({
    "Model": "Model C (TF-IDF Unigram + Linear SVM)",
    "Representation": "TF-IDF (Unigram)",
    "Classifier": "Linear SVM",
    "Vocabulary_Size": len(tfidf_unigram.vocabulary_),
    "Accuracy": round(acc_svm_uni, 4),
    "Precision": round(prec_svm_uni, 4),
    "Recall": round(rec_svm_uni, 4),
    "F1_Score": round(f1_svm_uni, 4),
    "Training_Time_Sec": round(svm_uni_train_time, 2)
})

print("--- 7. Experiment 5: n-gram features (Unigram vs Unigram + Bigram) ---")
t0 = time.time()
tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, preprocessor=clean_text_simple)
X_train_tfidf_bi = tfidf_bigram.fit_transform(X_train)
X_test_tfidf_bi = tfidf_bigram.transform(X_test)
tfidf_bi_fit_time = time.time() - t0

t1 = time.time()
clf_lr_bigram = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
clf_lr_bigram.fit(X_train_tfidf_bi, y_train)
lr_bi_train_time = time.time() - t1
total_bi_time = tfidf_bi_fit_time + lr_bi_train_time

y_pred_bigram = clf_lr_bigram.predict(X_test_tfidf_bi)
acc_bigram = accuracy_score(y_test, y_pred_bigram)
prec_bigram = precision_score(y_test, y_pred_bigram, pos_label='positive')
rec_bigram = recall_score(y_test, y_pred_bigram, pos_label='positive')
f1_bigram = f1_score(y_test, y_pred_bigram, pos_label='positive')

print(f"TF-IDF Uni+Bigram + LR -> Acc: {acc_bigram:.4f}, Prec: {prec_bigram:.4f}, Rec: {rec_bigram:.4f}, F1: {f1_bigram:.4f}, Time: {total_bi_time:.2f}s")
results_list.append({
    "Model": "Model D (TF-IDF Uni+Bigram + LR)",
    "Representation": "TF-IDF (Uni + Bigram)",
    "Classifier": "Logistic Regression",
    "Vocabulary_Size": len(tfidf_bigram.vocabulary_),
    "Accuracy": round(acc_bigram, 4),
    "Precision": round(prec_bigram, 4),
    "Recall": round(rec_bigram, 4),
    "F1_Score": round(f1_bigram, 4),
    "Training_Time_Sec": round(total_bi_time, 2)
})

# Linear SVM with Uni + Bigram
t0 = time.time()
clf_svm_bi = LinearSVC(random_state=SEED, max_iter=2000, C=1.0)
clf_svm_bi.fit(X_train_tfidf_bi, y_train)
svm_bi_train_time = time.time() - t0 + tfidf_bi_fit_time

y_pred_svm_bi = clf_svm_bi.predict(X_test_tfidf_bi)
acc_svm_bi = accuracy_score(y_test, y_pred_svm_bi)
prec_svm_bi = precision_score(y_test, y_pred_svm_bi, pos_label='positive')
rec_svm_bi = recall_score(y_test, y_pred_svm_bi, pos_label='positive')
f1_svm_bi = f1_score(y_test, y_pred_svm_bi, pos_label='positive')

print(f"TF-IDF Uni+Bigram + Linear SVM -> Acc: {acc_svm_bi:.4f}, Prec: {prec_svm_bi:.4f}, Rec: {rec_svm_bi:.4f}, F1: {f1_svm_bi:.4f}, Time: {svm_bi_train_time:.2f}s")
results_list.append({
    "Model": "Model E (TF-IDF Uni+Bigram + Linear SVM)",
    "Representation": "TF-IDF (Uni + Bigram)",
    "Classifier": "Linear SVM",
    "Vocabulary_Size": len(tfidf_bigram.vocabulary_),
    "Accuracy": round(acc_svm_bi, 4),
    "Precision": round(prec_svm_bi, 4),
    "Recall": round(rec_svm_bi, 4),
    "F1_Score": round(f1_svm_bi, 4),
    "Training_Time_Sec": round(svm_bi_train_time, 2)
})

# Save results.csv
results_df = pd.DataFrame(results_list)
results_df.to_csv('lab04/results.csv', index=False)
print("\n--- Benchmark Table (Saved to lab04/results.csv) ---")
print(results_df.to_string(index=False))

print("--- 8. Experiment 4: Word Order Demonstration ---")
s1 = "this movie is good"
s2 = "this movie is not good"

def get_doc_vector(s, w2v):
    tokens = s.lower().split()
    vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0)

v1 = get_doc_vector(s1, w2v_model)
v2 = get_doc_vector(s2, w2v_model)
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"S1: '{s1}'")
print(f"S2: '{s2}'")
print(f"Cosine similarity between S1 and S2 in Average Word2Vec: {cos_sim:.4f}")
print("Prediction on S1 with Average Word2Vec + LR:", clf_lr_w2v.predict([v1])[0])
print("Prediction on S2 with Average Word2Vec + LR:", clf_lr_w2v.predict([v2])[0])

# With TF-IDF Uni+Bigram:
v1_bi = tfidf_bigram.transform([s1])
v2_bi = tfidf_bigram.transform([s2])
print("Prediction on S1 with TF-IDF Uni+Bigram + LR:", clf_lr_bigram.predict(v1_bi)[0])
print("Prediction on S2 with TF-IDF Uni+Bigram + LR:", clf_lr_bigram.predict(v2_bi)[0])

print("--- 9. Confusion Matrix & Plotting ---")
cm = confusion_matrix(y_test, y_pred_bigram, labels=['positive', 'negative'])
print("Confusion Matrix for Best Model (TF-IDF Uni+Bigram + LR):")
print("Labels: [Positive, Negative]")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bigram, labels=['negative', 'positive']).ravel()
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
classes = ['Positive', 'Negative']
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix: TF-IDF Uni+Bigram + Logistic Regression\n(Accuracy: 90.07%, F1: 90.06%)',
       ylabel='Actual Label',
       xlabel='Predicted Label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        pct = val / cm.sum() * 100
        ax.text(j, i, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('lab04/confusion_matrix.png', dpi=300)
plt.close()
print("Saved lab04/confusion_matrix.png successfully!")

print("--- 10. Inspect Model Weights ---")
feature_names = np.array(tfidf_bigram.get_feature_names_out())
coefs = clf_lr_bigram.coef_[0]

# Positive class is mapped to 1 ('positive')
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_neg_idx = np.argsort(coefs)[:20]

top_pos_features = pd.DataFrame({
    'Feature': feature_names[top_pos_idx],
    'Weight': coefs[top_pos_idx]
})
top_neg_features = pd.DataFrame({
    'Feature': feature_names[top_neg_idx],
    'Weight': coefs[top_neg_idx]
})

print("\nTop 15 Positive Features (w_i >> 0):")
print(top_pos_features.head(15).to_string(index=False))

print("\nTop 15 Negative Features (w_i << 0):")
print(top_neg_features.head(15).to_string(index=False))

print("--- 11. Error Analysis Extraction ---")
test_df = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'pred_label': y_pred_bigram,
    'pred_prob_pos': clf_lr_bigram.predict_proba(X_test_tfidf_bi)[:, 1] if hasattr(clf_lr_bigram, "predict_proba") else 0
})

correct = test_df[test_df['true_label'] == test_df['pred_label']]
errors = test_df[test_df['true_label'] != test_df['pred_label']]
print(f"Total Correct: {len(correct)}, Total Errors: {len(errors)}")

# Save selected samples to inspect
correct.head(10).to_json('lab04/correct_samples.json', orient='records', indent=2)
errors.head(15).to_json('lab04/error_samples.json', orient='records', indent=2)

print("--- Script completed successfully! ---")
