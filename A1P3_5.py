from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split
from A1P3_4 import Word2vecModel
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def prepare_texts(text):
    # Get a callable object from spaCy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")

    # lemmatize the text, get part of speech, and remove spaces and punctuation

    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in [
        "PUNCT", "SPACE"]]

    # count the number of occurences of each word in the vocabulary

    freqs = Counter()
    for w in lemmas:
        freqs[w] += 1

    vocab = list(freqs.items())  # List of (word, occurrence)

    # Sort by decreasing frequency
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)
    print(vocab)

    # Create word->index dictionary and index->word dictionary

    v2i = {v[0]: i for i, v in enumerate(vocab)}
    i2v = {i: v[0] for i, v in enumerate(vocab)}

    return lemmas, v2i, i2v


def tokenize_and_preprocess_text(textlist, v2i, window=3):

    # Predict context with word. Sample the context within a window size.
    # based on the period at the end of the sentence, we'll split the text into sentences
    nlp = spacy.load("en_core_web_sm")
    sentence = textlist.split(".")
    X, Y = [], []  # is the list of training/test samples
    for i in sentence:
        lemmas = [tok.lemma_ for tok in nlp(i) if tok.pos_ not in [
            "PUNCT", "SPACE"]]
        for word in lemmas:
            # search each word within the window size from -(window-1)/2 to +(window-1)/2, not including the word itself
            for j in range(-(window-1)//2, (window-1)//2+1):
                if j != 0:
                    if lemmas.index(word)+j >= 0 and lemmas.index(word)+j < len(lemmas):
                        X.append(v2i[word])
                        Y.append(v2i[lemmas[lemmas.index(word)+j]])

    # TO DO - create all the X,Y pairs

    return np.array(X, dtype=int), np.array(Y, dtype=int)


def train_word2vec(textlist, window=5, embedding_size=2):
    # Set up a model with Skip-gram (predict context with word)
    # textlist: a list of the strings

    # Create the training data
    X, Y = tokenize_and_preprocess_text(textlist, v2i, window)
    # change Y to long type
    Y = np.array(Y).astype(np.int64)
    X, Y = np.array(X), np.array(Y)
    # Split the training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True)

    # instantiate the network & set up the optimizer
    network = Word2vecModel(len(v2i), embedding_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    epic = 50
    batch_size = 4
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # training loop
    bathes = torch.from_numpy(X_train).split(batch_size)
    targets = torch.from_numpy(Y_train).split(batch_size)
    #  Training plot the training loss and validation loss curve
    train_loss = []
    valid_loss = []
    for epoch in range(epic):
        epoch_loss = 0
        for x, y in zip(bathes, targets):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = network(x)
            loss = loss_fn(logits, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/len(bathes))
        # validation
        with torch.no_grad():
            logits, _ = network(torch.from_numpy(X_test).to(device))
            loss = loss_fn(logits, torch.from_numpy(Y_test).to(device))
            valid_loss.append(loss.item())
        print("Epoch: %d, Loss: %f" % (epoch, loss.item()))
    # plot the training loss and validation loss curve
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(valid_loss, label="Validation loss")
    plt.legend()
    plt.show()
    return network, train_loss, valid_loss
# def train_word2vec(textlist, window=5, embedding_size=2):
#     '''
#     Set up a model with Skip-gram (predict context with word)
#     textlist: a list of the strings
#     '''
#     # Create the training data
#     X, y = tokenize_and_preprocess_text(textlist, v2i) # moved to front for speed
#     y = y.astype(np.int64)
#     print (X.shape, y.shape)

#     # Split the training data

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#     # instantiate the network & set up the optimizer

#     model = Word2vecModel(vocab_size=len(v2i.keys()), embedding_size=embedding_size)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model.to(device)

#     lr = 1e-3
#     epochs = 50
#     bs = 4
#     n_workers = 1
#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

#     # training loop
#     batches = torch.from_numpy(X_train).split(bs)
#     targets = torch.from_numpy(y_train).split(bs)

#     progress_bar = tqdm(range(epochs))

#     running_loss = []
#     running_val_loss = []

#     for epoch in range(epochs):
#         epoch_loss = 0
#         for center, context in zip(batches, targets):
#             center, context = center.to(device), context.to(device)
#             optimizer.zero_grad()
#             logits, e = model(x=center) # forward
#             loss = loss_fn(logits, context)
#             epoch_loss += loss.item()
#             loss.backward()
#             optimizer.step()

#         val_pred, _ = model(x=torch.from_numpy(X_test))
#         val_loss = loss_fn(val_pred, torch.from_numpy(y_test)).item()

#         progress_bar.update(1)
#         epoch_loss /= len(batches)
#         running_loss.append(epoch_loss)
#         running_val_loss.append(val_loss)

#     return model, running_loss, running_val_loss


def visualize_embedding(embedding, most_frequent_from=0, most_frequent_to=40):
    assert embedding.shape[1] == 2, "This only supports visualizing 2-d embeddings!"
    # TO DO - visualize the embedding using matplotlib
    plt.figure()
    plt.scatter(embedding[most_frequent_from:most_frequent_to, 0],
                embedding[most_frequent_from:most_frequent_to, 1])
    for i in range(most_frequent_from, most_frequent_to):
        plt.annotate(i2v[i], (embedding[i, 0], embedding[i, 1]))
    plt.show()
    # TO DO


if __name__ == '__main__':

    np.random.seed(30)
    torch.manual_seed(30)
    text = open('SmallSimpleCorpus.txt').read()
    _, v2i, i2v = prepare_texts(text)
    network, running_loss, running_val_loss = train_word2vec(
        text, window=5, embedding_size=2)
    embedding = network.embeddings
    visualize_embedding(embedding.weight.data.numpy(),
                        most_frequent_from=0, most_frequent_to=11)
    # plot the training loss and validation loss curve
    # plt.figure()
    # plt.plot(running_loss, label="Training loss")
    # plt.plot(running_val_loss, label="Validation loss")
    # plt.legend()
    # plt.show()
