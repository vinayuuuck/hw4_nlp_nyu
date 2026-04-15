import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    synonym_prob = 0.12
    typo_prob = 0.08
    negation_words = {"not", "no", "never", "n't"}
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "these",
        "those", "i", "you", "we", "they", "my", "your", "our", "their", "me", "him", "her", "them"
    }
    qwerty_neighbors = {
        "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdfr", "f": "drtgvc", "g": "ftyhbv",
        "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm",
        "o": "iklp", "p": "ol", "q": "wa", "r": "edfgt", "s": "awedxz", "t": "rfghy", "u": "yhjki",
        "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"
    }

    def preserve_case(original, replacement):
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement

    def eligible_token(token):
        lower = token.lower()
        if not token.isalpha():
            return False
        if len(token) < 3:
            return False
        if lower in negation_words:
            return False
        if lower in stopwords:
            return False
        return True

    def replace_with_synonym(token):
        if (not eligible_token(token)) or random.random() >= synonym_prob:
            return token

        lower = token.lower()
        candidates = set()
        for synset in wordnet.synsets(lower):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower().strip()
                if candidate == lower:
                    continue
                if " " in candidate or "-" in candidate:
                    continue
                if not candidate.isalpha():
                    continue
                candidates.add(candidate)

        if not candidates:
            return token

        replacement = random.choice(list(candidates))
        return preserve_case(token, replacement)

    def inject_typo(token):
        if (not eligible_token(token)) or random.random() >= typo_prob:
            return token

        candidate_positions = [i for i, ch in enumerate(token) if ch.lower() in qwerty_neighbors]
        if not candidate_positions:
            return token

        idx = random.choice(candidate_positions)
        original_char = token[idx]
        neighbor_chars = qwerty_neighbors[original_char.lower()]
        replacement_char = random.choice(neighbor_chars)
        if original_char.isupper():
            replacement_char = replacement_char.upper()

        return token[:idx] + replacement_char + token[idx + 1:]

    tokens = word_tokenize(example["text"])
    transformed_tokens = []

    for token in tokens:
        transformed = replace_with_synonym(token)
        transformed = inject_typo(transformed)
        transformed_tokens.append(transformed)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
