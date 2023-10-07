from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm  # For progress bars

def prepare_texts(text, min_frequency=3):
    
    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences 
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    
    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    
    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case
    
    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency, vocab))
    # print(frequent_vocab)
    # Create the dictionaries to go from word to index or vice-verse
    
    w2i = {w[0]:i for i,w in enumerate(frequent_vocab)}
    i2w = {i:w[0] for i,w in enumerate(frequent_vocab)}
    
    # Create an Out Of Vocabulary (oov) token as well
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"
    
    # Set all of the words not included in vocabulary nuas oov
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)
    
    return filtered_lemmas, w2i, i2w, vocab


def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    n_grams = (window - 1) // 2
    progress_bar = tqdm(range(len(textlist)))
    nlp = spacy.load("en_core_web_sm")

    for stc in textlist:
        lemma = []
        for sent in sent_tokenize(stc):  # sent_tokenize separates the sentences 
            for tok in nlp(sent):         # nlp processes as in Part III
                if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                    lemma.append(tok.lemma_)
        lemma = [w2i[i] if i in w2i else 2559 for i in lemma] # transfer to indices
        for i, w in enumerate(lemma):
            for n in range(1, n_grams + 1):
                if i - n >= 0:
                    # positive sampling
                    X.append(w)
                    T.append(lemma[i - n])
                    Y.append(1)
                    # negative sampling
                    X.append(w)
                    T.append(np.random.randint(0, 2560))
                    Y.append(0)
                if i + n < len(lemma):
                    # positive sampling
                    X.append(w)
                    T.append(lemma[i + n])
                    Y.append(1)
                    # negative sampling
                    X.append(w)
                    T.append(np.random.randint(0, 2560))
                    Y.append(0)

        progress_bar.update(1)
    return X, T, Y


if __name__ == "__main__":
    with open('LargerCorpus.txt','r', encoding='UTF-8') as f:
        txt = f.read()
    textlist = sent_tokenize(txt)
    filtered_lemmas, w2i, i2w, vocab = prepare_texts(txt)
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, 5)
    print(len(X))
    print("first 20 tokens: ", [i2w[i] for i in X[:20]])
    print("first 20 context tokens: ", [i2w[i] for i in T[:20]])
    print("first 20 labels: ", Y[:20])