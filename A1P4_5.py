from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.model_selection import train_test_split
from A1P3_4 import Word2vecModel
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# prepare text using the both the nltk sentence tokenizer (https://www.nltk.org/api/nltk.tokenize.html)
# AND the spacy english pipeline (see https://spacy.io/models/en)


def prepare_texts(text, min_frequency=3):
    
    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences 
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)
    #print("Number of lemmas: ", len(lemmas))
    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    
    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case
    
    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency, vocab))

    #print("Number of frequent lemmas: ", len(frequent_vocab))
    #print ("Top 20 most frequent lemmas: ", frequent_vocab[:20])
    
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
    
    return filtered_lemmas, w2i, i2w

def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    # Tokenize the input
    nlp = spacy.load("en_core_web_sm")
    #  set progress bar
    sent_list = sent_tokenize(textlist)
    progress_bar = tqdm(range(len(sent_list)))

    for sent in sent_list:
        lemmas = []
        for tok in nlp(sent):
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)
    # convert words to indices if the word is in the vocabulary, otherwise convert to <oov>
        lemmas = [w2i[i] if i in w2i else w2i['<oov>'] for i in lemmas]
    # Loop through each token
        for token in lemmas:
            # if the word or the context is <oov>, skip it
            if token == w2i['<oov>']:
                continue
            # search each word within the window size from -(window-1)/2 to +(window-1)/2, not including the word itself
            for j in range(-(window-1)//2, (window-1)//2+1):
                if j != 0:
                    #postive sampling
                    if lemmas.index(token)+j >= 0 and lemmas.index(token)+j < len(lemmas):
                        if lemmas[lemmas.index(token)+j] == w2i['<oov>']:
                            continue
                        X.append(token)
                        T.append(lemmas[lemmas.index(token)+j])
                        Y.append(1)
                    #negative sampling
                        X.append(token)
                        T.append(np.random.randint(0, len(w2i)))
                        Y.append(0)

        progress_bar.update(1)
        
    return X, T, Y


if __name__ == "__main__":
    with open('LargerCorpus.txt', 'r', encoding='UTF-8') as f:
        txt = f.read()
    lemmas, w2i, i2w = prepare_texts(txt)
    X, T, Y = tokenize_and_preprocess_text(txt, w2i, 5)
   # convert the first 20 tokens back to words
    print("First 20 tokens: ", [i2w[i] for i in X[:20]])
    print("First 20 context tokens: ", [i2w[i] for i in T[:20]])
    print ("First 20 labels: ", Y[:20])
    print("Number of training samples: ", len(X))