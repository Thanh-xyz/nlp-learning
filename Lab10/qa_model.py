import re
import string
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    "that", "this", "these", "those", "it", "its", "and", "or", "but", "not"
}

def normalize_answer(text):
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)

def evaluate_qa_predictions(predictions, gold_answers_list):
    em_scores = []
    f1_scores = []
    for pred, gold_list in zip(predictions, gold_answers_list):
        if not gold_list:
            gold_list = [""]
        best_em = max(compute_exact_match(pred, g) for g in gold_list)
        best_f1 = max(compute_token_f1(pred, g) for g in gold_list)
        em_scores.append(best_em)
        f1_scores.append(best_f1)
    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "em": mean_em,
        "f1": mean_f1,
        "per_sample_em": em_scores,
        "per_sample_f1": f1_scores
    }

class KeywordOverlapQABaseline:
    def tokenize_words(self, text):
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def split_sentences(self, context):
        sentences = re.split(r"(?<=[.!?])\s+", context.strip())
        return [s for s in sentences if s]

    def predict(self, question, context):
        q_tokens = set(self.tokenize_words(question)) - STOPWORDS
        if not q_tokens:
            q_tokens = set(self.tokenize_words(question))
        sentences = self.split_sentences(context)
        if not sentences:
            return ""
        best_sentence = sentences[0]
        best_score = -1.0
        for sent in sentences:
            s_tokens = set(self.tokenize_words(sent))
            overlap = len(q_tokens & s_tokens)
            score = overlap / (len(s_tokens) + 1e-6) + 0.1 * overlap
            if score > best_score:
                best_score = score
                best_sentence = sent

        raw_words = best_sentence.rstrip(".!?").split()
        filtered = [w for w in raw_words if w.lower().strip(string.punctuation) not in q_tokens and w.lower().strip(string.punctuation) not in STOPWORDS]
        if not filtered:
            return best_sentence.rstrip(".!?")
        span_limit = min(4, len(filtered))
        return " ".join(filtered[:span_limit])

class ExtractiveQAModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained_qa_weights=None):
        super().__init__()
        if pretrained_qa_weights is not None:
            self.encoder = pretrained_qa_weights.base_model
            self.qa_outputs = pretrained_qa_weights.qa_outputs
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None
    ):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits
        }

def decode_best_span(
    start_logits_list,
    end_logits_list,
    offsets_list,
    sequence_ids_list,
    context_text,
    n_best_size=20,
    max_answer_length=30
):
    best_score = -1e9
    best_answer = ""
    best_indices = (0, 0)

    for start_logits, end_logits, offsets, seq_ids in zip(
        start_logits_list, end_logits_list, offsets_list, sequence_ids_list
    ):
        start_indexes = sorted(range(len(start_logits)), key=lambda i: start_logits[i], reverse=True)[:n_best_size]
        end_indexes = sorted(range(len(end_logits)), key=lambda i: end_logits[i], reverse=True)[:n_best_size]

        for start_index in start_indexes:
            if start_index >= len(offsets) or seq_ids[start_index] != 1:
                continue
            if offsets[start_index] == (0, 0):
                continue
            for end_index in end_indexes:
                if end_index >= len(offsets) or seq_ids[end_index] != 1:
                    continue
                if offsets[end_index] == (0, 0):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                score = start_logits[start_index] + end_logits[end_index]
                if score > best_score:
                    best_score = score
                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    best_answer = context_text[start_char:end_char]
                    best_indices = (start_index, end_index)

    return best_answer, best_score, best_indices
