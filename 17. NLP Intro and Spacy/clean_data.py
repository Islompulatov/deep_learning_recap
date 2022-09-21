import string
import numpy as np


def preprocess(pth):
    ponctuations = string.punctuation
    preprocessed = []
    with open(pth, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for ponct in ponctuations:
            line = line.replace(ponct, '').lower()
        preprocessed.append(line)
    return preprocessed


def clean_sentences(sentences):
    words_in_sent = [sent.split() for sent in sentences]
    for sent in words_in_sent:
        if len(sent) != 0:
            continue
        else:
            words_in_sent.remove(sent)

    return words_in_sent


def get_dicts(sentences):
    vocab = set()
    for sent in sentences:
        for word in sent:
            vocab.add(word)

    return {word: idx for idx, word in enumerate(list(vocab))}, len(vocab)


def get_pairs(sentences, word_to_idx, r):
    pairs = []
    for sent in sentences:
        tokens = [word_to_idx[word] for word in sent]

        for center in range(len(tokens)):
            for context in range(-r, r+1):
                context_word = center + context

                if context_word < 0 or context_word >= len(tokens) or context_word == center:
                    continue
                else:
                    pairs.append((tokens[center], tokens[context_word]))

    return np.array(pairs)