import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics.sequence_labeling import get_entities

TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

class InformationExtractor:
    def __init__(self, model_name='dslim/bert-base-NER'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict_tokens(self, words):
        if not words:
            return []
        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=128)
        word_ids = inputs.word_ids(0)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        predicted_tags = []
        previous_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                predicted_tags.append(self.model.config.id2label[preds[i]])
                previous_word_id = word_id
        return predicted_tags

    def extract_structured_entities(self, text_or_words):
        if isinstance(text_or_words, str):
            words = text_or_words.strip().split()
        else:
            words = text_or_words

        tags = self.predict_tokens(words)
        entities_list = get_entities(tags)
        structured_data = {
            'PERSON': [],
            'ORG': [],
            'LOCATION': []
        }
        for ent_type, start, end in entities_list:
            entity_text = ' '.join(words[start:end + 1])
            if ent_type == 'PER':
                structured_data['PERSON'].append(entity_text)
            elif ent_type == 'ORG':
                structured_data['ORG'].append(entity_text)
            elif ent_type == 'LOC':
                structured_data['LOCATION'].append(entity_text)

        for key in structured_data:
            structured_data[key] = list(dict.fromkeys(structured_data[key]))
        return structured_data

def evaluate_ie_on_dataset(extractor, test_data, num_sentences=30):
    selected_samples = test_data.select(range(num_sentences))
    type_map = {'PER': 'PERSON', 'ORG': 'ORG', 'LOC': 'LOCATION'}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_type_metrics = {
        'PERSON': {'tp': 0, 'fp': 0, 'fn': 0},
        'ORG': {'tp': 0, 'fp': 0, 'fn': 0},
        'LOCATION': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    for sample in selected_samples:
        words = sample['tokens']
        gold_tags = [TAG_NAMES[t] for t in sample['ner_tags']]
        gold_entities_raw = get_entities(gold_tags)

        gold_entities = set()
        for ent_type, start, end in gold_entities_raw:
            if ent_type in type_map:
                mapped_type = type_map[ent_type]
                text = ' '.join(words[start:end + 1])
                gold_entities.add((mapped_type, text))

        pred_structured = extractor.extract_structured_entities(words)
        pred_entities = set()
        for k, v_list in pred_structured.items():
            for text in v_list:
                pred_entities.add((k, text))

        for ent in gold_entities:
            ent_type, text = ent
            if ent in pred_entities:
                total_tp += 1
                per_type_metrics[ent_type]['tp'] += 1
            else:
                total_fn += 1
                per_type_metrics[ent_type]['fn'] += 1

        for ent in pred_entities:
            ent_type, text = ent
            if ent not in gold_entities:
                total_fp += 1
                per_type_metrics[ent_type]['fp'] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'total_evaluated_sentences': num_sentences,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_type': per_type_metrics
    }

def main():
    extractor = InformationExtractor()
    sample_text = 'Barack Obama visited Paris yesterday to speak with representatives of Google .'
    extracted_json = extractor.extract_structured_entities(sample_text)
    print('=== DEMO INFORMATION EXTRACTION ===')
    print('Raw Text:', sample_text)
    print('Extracted Entities (JSON):')
    print(json.dumps(extracted_json, indent=2, ensure_ascii=False))

    dataset = load_dataset('lhoestq/conll2003')
    eval_results = evaluate_ie_on_dataset(extractor, dataset['test'], num_sentences=30)
    print('\n=== INFORMATION EXTRACTION EVALUATION (30 GOLD SENTENCES) ===')
    print(f'Sentences Evaluated: {eval_results["total_evaluated_sentences"]}')
    print(f'Entity Precision   : {eval_results["precision"]:.4f}')
    print(f'Entity Recall      : {eval_results["recall"]:.4f}')
    print(f'Entity F1 Score    : {eval_results["f1"]:.4f}')
    print('\nPer-Entity Breakdown:')
    for ent_type, stats in eval_results['per_type'].items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_type = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f'- {ent_type:8s} | TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d} | P: {p:.4f} | R: {r:.4f} | F1: {f1_type:.4f}')

if __name__ == '__main__':
    main()
