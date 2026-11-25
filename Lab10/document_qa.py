import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from qa_model import decode_best_span, evaluate_qa_predictions

class DocumentQASystem:
    def __init__(self, model_name="distilbert-base-cased-distilled-squad", max_length=384, stride=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def answer_question(self, question, document):
        encoding = self.tokenizer(
            question,
            document,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        offsets_tensor = encoding.pop("offset_mapping")
        encoding.pop("overflow_to_sample_mapping")
        num_windows = encoding["input_ids"].size(0)
        seq_ids_list = [encoding.sequence_ids(i) for i in range(num_windows)]
        offsets_list = [offsets_tensor[i].tolist() for i in range(num_windows)]

        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits_list = outputs.start_logits.cpu().tolist()
            end_logits_list = outputs.end_logits.cpu().tolist()

        answer_text, best_score, span_tokens = decode_best_span(
            start_logits_list,
            end_logits_list,
            offsets_list,
            seq_ids_list,
            document
        )
        start_char = document.find(answer_text) if answer_text else -1
        end_char = start_char + len(answer_text) if start_char >= 0 else -1
        return {
            "question": question,
            "answer": answer_text,
            "score": round(best_score, 4),
            "char_span": [start_char, end_char],
            "num_windows": num_windows
        }

def build_demo_corpus():
    return [
        {
            "title": "Transformer Architecture",
            "context": (
                "The Transformer is a deep learning architecture introduced in 2017 by researchers at Google "
                "in the landmark paper Attention Is All You Need. Unlike recurrent neural networks such as LSTM, "
                "the Transformer relies entirely on self-attention mechanisms to draw global dependencies between "
                "input and output without sequential recurrence. Vaswani et al. demonstrated that this design "
                "significantly improves parallelization during training and achieves state-of-the-art results "
                "on machine translation benchmarks including WMT 2014 English-to-German."
            ),
            "qa_pairs": [
                ("When was the Transformer architecture introduced?", "2017"),
                ("Which company's researchers introduced the Transformer?", "Google"),
                ("What paper introduced the Transformer?", "Attention Is All You Need"),
                ("What mechanism does the Transformer rely on instead of recurrence?", "self-attention mechanisms")
            ]
        },
        {
            "title": "Extractive Question Answering and SQuAD",
            "context": (
                "The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting "
                "of questions posed by crowdworkers on a set of Wikipedia articles. In SQuAD 1.1, the answer "
                "to every question is a segment of text, or span, from the corresponding reading passage. "
                "SQuAD contains over 100,000 question-answer pairs on more than 500 articles. Standard "
                "evaluation uses two metrics: Exact Match (EM) and Token-level F1 score."
            ),
            "qa_pairs": [
                ("Who posed the questions in SQuAD?", "crowdworkers"),
                ("What source of articles does SQuAD use?", "Wikipedia"),
                ("How many question-answer pairs does SQuAD contain?", "over 100,000"),
                ("What are the two standard evaluation metrics for SQuAD?", "Exact Match (EM) and Token-level F1 score")
            ]
        },
        {
            "title": "Hanoi University of Science FAQ",
            "context": (
                "VNU University of Science (HUS) was established in 1995 as a member institution of Vietnam "
                "National University, Hanoi, tracing its roots back to Indochina University founded in 1906. "
                "The main campus is located at 334 Nguyen Trai Street in Thanh Xuan District, Hanoi. "
                "In the Natural Language Processing course, students progress from classical embeddings in Lab 3 "
                "to BERT fine-tuning in Lab 8, Named Entity Recognition in Lab 9, and Extractive QA in Lab 10."
            ),
            "qa_pairs": [
                ("When was VNU University of Science established?", "1995"),
                ("Where is the main campus of HUS located?", "334 Nguyen Trai Street in Thanh Xuan District, Hanoi"),
                ("Which lab covers Named Entity Recognition?", "Lab 9"),
                ("What year was Indochina University founded?", "1906")
            ]
        }
    ]

def main():
    qa_system = DocumentQASystem()
    corpus = build_demo_corpus()

    predictions = []
    references = []
    results_log = []

    for doc in corpus:
        for question, gold_answer in doc["qa_pairs"]:
            result = qa_system.answer_question(question, doc["context"])
            predictions.append(result["answer"])
            references.append([gold_answer])
            results_log.append({
                "domain": doc["title"],
                "question": question,
                "gold": gold_answer,
                "prediction": result["answer"],
                "score": result["score"],
                "char_span": result["char_span"]
            })

    metrics = evaluate_qa_predictions(predictions, references)
    print("=== MINI DOCUMENT QA APPLICATION RESULTS ===")
    for idx, row in enumerate(results_log, start=1):
        print(f"[{idx:02d}] ({row['domain']}) Q: {row['question']}")
        print(f"     Pred: {row['prediction']} | Gold: {row['gold']} | Score: {row['score']} | Span: {row['char_span']}")

    print("\n=== QUANTITATIVE EVALUATION ON DOMAIN CORPUS ===")
    print(f"Total Questions : {len(predictions)}")
    print(f"Exact Match (EM): {metrics['em']:.4f}")
    print(f"Token-level F1  : {metrics['f1']:.4f}")
    print("\nSample JSON Output:")
    print(json.dumps(results_log[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
