import math
import re
from collections import Counter


def tokenize(text):
    if isinstance(text, list):
        return text
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def build_vocabulary(corpus, min_freq=1, unk_token="<unk>"):
    counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        counts.update(tokens)
    vocab = [token for token, freq in counts.items() if freq >= min_freq]
    if unk_token and unk_token not in vocab:
        vocab.append(unk_token)
    return sorted(list(set(vocab)))


def count_ngrams(corpus, n=1, pad_left=False, pad_right=False, sos="<s>", eos="</s>"):
    counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        padded = list(tokens)
        if pad_left:
            padded = [sos] * (n - 1) + padded
        if pad_right:
            padded = padded + [eos]
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            counts[tuple(padded[i:i + n])] += 1
    return counts


class NGramLanguageModel:
    def __init__(self, n=2, smoothing=None, delta=1.0, pad_left=True, pad_right=True, sos="<s>", eos="</s>", unk_token="<unk>"):
        self.n = n
        self.smoothing = smoothing.lower() if smoothing else "mle"
        self.delta = delta
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.sos = sos
        self.eos = eos
        self.unk_token = unk_token
        self.vocab = set()
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.total_tokens = 0
        self.vocab_size = 0

    def _map_token(self, token):
        if self.unk_token and token not in self.vocab:
            return self.unk_token
        return token

    def fit(self, corpus, min_freq=1):
        raw_tokenized = [tokenize(doc) for doc in corpus]
        token_freq = Counter()
        for toks in raw_tokenized:
            token_freq.update(toks)

        self.vocab = set()
        for token, count in token_freq.items():
            if count >= min_freq:
                self.vocab.add(token)

        if self.unk_token:
            self.vocab.add(self.unk_token)
        if self.pad_right:
            self.vocab.add(self.eos)

        self.vocab_size = len(self.vocab)
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.total_tokens = 0

        for toks in raw_tokenized:
            mapped_tokens = [self._map_token(w) for w in toks]
            for w in mapped_tokens:
                self.unigram_counts[w] += 1
                self.total_tokens += 1

            padded = mapped_tokens
            if self.pad_left:
                padded = [self.sos] * (self.n - 1) + padded
            if self.pad_right:
                padded = padded + [self.eos]

            for i in range(len(padded)):
                w = padded[i]
                if self.n == 1:
                    self.ngram_counts[(w,)] += 1
                if i < len(padded) - 1:
                    bg = (padded[i], padded[i + 1])
                    self.bigram_counts[bg] += 1
                if self.n > 1 and i <= len(padded) - self.n:
                    ng = tuple(padded[i:i + self.n])
                    ctx = ng[:-1]
                    self.ngram_counts[ng] += 1
                    self.context_counts[ctx] += 1
        return self

    def probability(self, context, word):
        if isinstance(context, str):
            context = (context,) if context else ()
        elif isinstance(context, list):
            context = tuple(context)

        word = self._map_token(word)
        context = tuple(self._map_token(w) if w != self.sos else w for w in context)
        v = self.vocab_size

        if len(context) == 0 or self.n == 1:
            c_w = self.unigram_counts[word]
            if self.smoothing in ("laplace", "add-one"):
                return (c_w + 1.0) / (self.total_tokens + v) if (self.total_tokens + v) > 0 else 0.0
            elif self.smoothing in ("add-k", "lidstone"):
                return (c_w + self.delta) / (self.total_tokens + self.delta * v) if (self.total_tokens + self.delta * v) > 0 else 0.0
            else:
                return (c_w / self.total_tokens) if self.total_tokens > 0 else 0.0

        if self.n == 3 and len(context) == 1:
            c_bg = self.bigram_counts[(context[0], word)]
            c_ctx = self.unigram_counts[context[0]]
            if self.smoothing in ("laplace", "add-one"):
                return (c_bg + 1.0) / (c_ctx + v) if (c_ctx + v) > 0 else (1.0 / v if v > 0 else 0.0)
            elif self.smoothing in ("add-k", "lidstone"):
                return (c_bg + self.delta) / (c_ctx + self.delta * v) if (c_ctx + self.delta * v) > 0 else (1.0 / v if v > 0 else 0.0)
            else:
                return (c_bg / c_ctx) if c_ctx > 0 else 0.0

        if len(context) > self.n - 1:
            context = context[-(self.n - 1):]
        elif len(context) < self.n - 1 and self.pad_left:
            context = tuple([self.sos] * (self.n - 1 - len(context)) + list(context))

        ngram = context + (word,)
        c_ngram = self.ngram_counts[ngram]
        c_ctx = self.context_counts[context]

        if self.smoothing in ("laplace", "add-one"):
            return (c_ngram + 1.0) / (c_ctx + v) if (c_ctx + v) > 0 else (1.0 / v if v > 0 else 0.0)
        elif self.smoothing in ("add-k", "lidstone"):
            return (c_ngram + self.delta) / (c_ctx + self.delta * v) if (c_ctx + self.delta * v) > 0 else (1.0 / v if v > 0 else 0.0)
        else:
            return (c_ngram / c_ctx) if c_ctx > 0 else 0.0

    def sentence_probability(self, sentence):
        raw_tokens = tokenize(sentence)
        if not raw_tokens:
            return 0.0
        tokens = [self._map_token(w) for w in raw_tokens]
        prob = 1.0
        if self.pad_left:
            padded = [self.sos] * (self.n - 1) + tokens + ([self.eos] if self.pad_right else [])
            for i in range(self.n - 1, len(padded)):
                w = padded[i]
                ctx = tuple(padded[i - (self.n - 1):i])
                prob *= self.probability(ctx, w)
        else:
            for i in range(len(tokens)):
                w = tokens[i]
                ctx = tuple(tokens[max(0, i - (self.n - 1)):i])
                prob *= self.probability(ctx, w)
        return prob

    def sentence_log_probability(self, sentence):
        raw_tokens = tokenize(sentence)
        if not raw_tokens:
            return float("-inf")
        tokens = [self._map_token(w) for w in raw_tokens]
        log_prob = 0.0
        if self.pad_left:
            padded = [self.sos] * (self.n - 1) + tokens + ([self.eos] if self.pad_right else [])
            for i in range(self.n - 1, len(padded)):
                w = padded[i]
                ctx = tuple(padded[i - (self.n - 1):i])
                p = self.probability(ctx, w)
                if p <= 0.0:
                    return float("-inf")
                log_prob += math.log(p)
        else:
            for i in range(len(tokens)):
                w = tokens[i]
                ctx = tuple(tokens[max(0, i - (self.n - 1)):i])
                p = self.probability(ctx, w)
                if p <= 0.0:
                    return float("-inf")
                log_prob += math.log(p)
        return log_prob

    def perplexity(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        total_log_prob = 0.0
        total_tokens = 0
        for sent in corpus:
            tokens = tokenize(sent)
            if not tokens:
                continue
            num_tokens = len(tokens) + (1 if self.pad_right else 0)
            log_p = self.sentence_log_probability(sent)
            if math.isinf(log_p) and log_p < 0:
                return float("inf")
            total_log_prob += log_p
            total_tokens += num_tokens
        if total_tokens == 0:
            return float("inf")
        return math.exp(- total_log_prob / total_tokens)

    def next_word_distribution(self, context):
        dist = {}
        for word in self.vocab:
            if word in (self.sos, self.unk_token):
                continue
            dist[word] = self.probability(context, word)
        sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        return sorted_dist

    def generate(self, context=None, max_tokens=20, stop_token="</s>"):
        tokens = list(tokenize(context)) if context else []
        for _ in range(max_tokens):
            cur_ctx = tuple(tokens[-(self.n - 1):]) if self.n > 1 else ()
            dist = self.next_word_distribution(cur_ctx)
            if not dist or dist[0][1] <= 0:
                break
            next_word = dist[0][0]
            if next_word == stop_token:
                break
            tokens.append(next_word)
        return " ".join(tokens)


def train_unigram(corpus, smoothing=None):
    model = NGramLanguageModel(n=1, smoothing=smoothing, pad_left=False, pad_right=False)
    model.fit(corpus)
    return model


def train_bigram(corpus, smoothing=None, pad_left=True, pad_right=True):
    model = NGramLanguageModel(n=2, smoothing=smoothing, pad_left=pad_left, pad_right=pad_right)
    model.fit(corpus)
    return model


def train_trigram(corpus, smoothing=None, pad_left=True, pad_right=True):
    model = NGramLanguageModel(n=3, smoothing=smoothing, pad_left=pad_left, pad_right=pad_right)
    model.fit(corpus)
    return model


def probability(context, word, model):
    return model.probability(context, word)


def sentence_probability(sentence, model):
    return model.sentence_probability(sentence)


def sentence_log_probability(sentence, model):
    return model.sentence_log_probability(sentence)


def run_unit_tests():
    corpus = [
        "the cat eats fish",
        "the cat likes fish",
        "the dog eats meat"
    ]

    vocab = build_vocabulary(corpus, unk_token=None)
    assert len(vocab) == 7
    assert set(vocab) == {"the", "cat", "dog", "eats", "fish", "likes", "meat"}

    unigram = NGramLanguageModel(n=1, smoothing=None, pad_left=False, pad_right=False, unk_token=None)
    unigram.fit(corpus)
    assert math.isclose(unigram.probability((), "the"), 3 / 12)
    assert math.isclose(unigram.probability((), "cat"), 2 / 12)
    assert math.isclose(unigram.probability((), "fish"), 2 / 12)
    assert math.isclose(unigram.probability((), "dog"), 1 / 12)

    total_unigram_p = sum(unigram.probability((), w) for w in vocab)
    assert math.isclose(total_unigram_p, 1.0)

    bigram_unpadded = NGramLanguageModel(n=2, smoothing=None, pad_left=False, pad_right=False, unk_token=None)
    bigram_unpadded.fit(corpus)
    assert math.isclose(bigram_unpadded.probability("the", "cat"), 2 / 3)
    assert math.isclose(bigram_unpadded.probability("the", "dog"), 1 / 3)
    assert math.isclose(bigram_unpadded.probability("cat", "eats"), 1 / 2)
    assert math.isclose(bigram_unpadded.probability("cat", "likes"), 1 / 2)

    p_sentence = bigram_unpadded.sentence_probability("the cat eats fish")
    expected_p = (3 / 12) * (2 / 3) * (1 / 2) * (1 / 2)
    assert math.isclose(p_sentence, expected_p)
    assert math.isclose(p_sentence, 1 / 24)

    log_p_sentence = bigram_unpadded.sentence_log_probability("the cat eats fish")
    assert math.isclose(math.exp(log_p_sentence), 1 / 24)

    dummy_model = NGramLanguageModel(n=2, smoothing="laplace", pad_left=False, pad_right=False, unk_token=None)
    dummy_model.vocab = {"cat", "dog", "eats", "fish", "meat"}
    dummy_model.vocab_size = 5
    dummy_model.context_counts[("cat",)] = 10
    dummy_model.ngram_counts[("cat", "eats")] = 0
    p_laplace_zero = dummy_model.probability("cat", "eats")
    assert math.isclose(p_laplace_zero, 1 / 15)

    dummy_model.ngram_counts[("cat", "eats")] = 3
    p_laplace_three = dummy_model.probability("cat", "eats")
    assert math.isclose(p_laplace_three, 4 / 15)

    probs = [0.5, 0.25, 0.5]
    p_seq = math.prod(probs)
    pp_seq = p_seq ** (-1 / 3)
    assert math.isclose(p_seq, 0.0625)
    assert math.isclose(pp_seq, 16 ** (1 / 3))

    probs2 = [0.5, 0.1, 0.5]
    p_seq2 = math.prod(probs2)
    pp_seq2 = p_seq2 ** (-1 / 3)
    assert math.isclose(p_seq2, 0.025)
    assert math.isclose(pp_seq2, 40 ** (1 / 3))

    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_unit_tests()
