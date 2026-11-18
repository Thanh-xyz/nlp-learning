import os
import csv
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report, f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r

TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def extract_token_features(sentence, index):
    word = sentence[index]
    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.length': len(word),
    }
    if index > 0:
        prev_word = sentence[index - 1]
        features.update({
            '-1:word.lower': prev_word.lower(),
            '-1:word.istitle': prev_word.istitle(),
            '-1:word.isupper': prev_word.isupper(),
        })
    else:
        features['BOS'] = True

    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features.update({
            '+1:word.lower': next_word.lower(),
            '+1:word.istitle': next_word.istitle(),
            '+1:word.isupper': next_word.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def prepare_dataset(split_data, sample_limit=None):
    if sample_limit is not None:
        split_data = split_data.select(range(min(sample_limit, len(split_data))))
    feature_dicts = []
    labels = []
    for tokens, tags in zip(split_data['tokens'], split_data['ner_tags']):
        if len(tokens) == 0:
            continue
        for i in range(len(tokens)):
            feature_dicts.append(extract_token_features(tokens, i))
            labels.append(TAG_NAMES[tags[i]])
    return feature_dicts, labels

def evaluate_baseline(model, vectorizer, test_data, sample_limit=1000):
    if sample_limit is not None:
        test_data = test_data.select(range(min(sample_limit, len(test_data))))
    gold_sequences = []
    predicted_sequences = []
    for tokens, tags in zip(test_data['tokens'], test_data['ner_tags']):
        if len(tokens) == 0:
            continue
        gold_tags = [TAG_NAMES[t] for t in tags]
        token_features = [extract_token_features(tokens, i) for i in range(len(tokens))]
        feature_matrix = vectorizer.transform(token_features)
        predictions = list(model.predict(feature_matrix))
        gold_sequences.append(gold_tags)
        predicted_sequences.append(predictions)

    flat_gold = [t for s in gold_sequences for t in s]
    flat_pred = [t for s in predicted_sequences for t in s]

    token_acc = accuracy_score(flat_gold, flat_pred)
    token_f1 = f1_score(flat_gold, flat_pred, average='macro')
    entity_p = seq_p(gold_sequences, predicted_sequences)
    entity_r = seq_r(gold_sequences, predicted_sequences)
    entity_f = seq_f1(gold_sequences, predicted_sequences)

    return {
        'token_accuracy': token_acc,
        'token_macro_f1': token_f1,
        'entity_precision': entity_p,
        'entity_recall': entity_r,
        'entity_f1': entity_f,
        'classification_report': classification_report(gold_sequences, predicted_sequences),
        'gold_sequences': gold_sequences,
        'predicted_sequences': predicted_sequences
    }

def main():
    dataset = load_dataset('lhoestq/conll2003')
    train_features, train_labels = prepare_dataset(dataset['train'], sample_limit=5000)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    classifier = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs')
    classifier.fit(x_train, train_labels)

    metrics = evaluate_baseline(classifier, vectorizer, dataset['test'], sample_limit=1000)

    print('=== BASELINE NER EVALUATION RESULTS ===')
    print(f'Token-level Accuracy : {metrics["token_accuracy"]:.4f}')
    print(f'Token-level Macro F1 : {metrics["token_macro_f1"]:.4f}')
    print(f'Entity-level Precision: {metrics["entity_precision"]:.4f}')
    print(f'Entity-level Recall   : {metrics["entity_recall"]:.4f}')
    print(f'Entity-level F1       : {metrics["entity_f1"]:.4f}')
    print('\nEntity-level Classification Report:')
    print(metrics['classification_report'])

if __name__ == '__main__':
    main()
