import numpy as np
import torch
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
import os

from A1P4_5 import *
from A1P4_6 import SkipGramNegativeSampling

def train_sgns(textlist, window, embedding_size):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # textlist: a list of strings
    
    # Create Training Data 
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    # convert data to numpy array
    X, T, Y = np.array(X), np.array(T), np.array(Y)


    # Split the training data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, shuffle=True)
    # instantiate the network & set up the optimizer
    network = SkipGramNegativeSampling(len(w2i), embedding_size)
    model = network.to(device)
    lr = 1e-3
    bs = 4
    epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # training loop
    batches = torch.from_numpy(X_train).split(bs)
    targets = torch.from_numpy(T_train).split(bs)
    labels = torch.from_numpy(Y_train).split(bs)

    progress_bar = tqdm(range(epochs))

    running_loss = []
    running_val_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for center, context, label in zip(batches, targets, labels):
            center, context, label = center.to(device), context.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(x=center, t=context)
            loss = loss_fn(logits, label.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
    # validation
        with torch.no_grad():
            val_pred = model(x=torch.from_numpy(X_test), t=torch.from_numpy(T_test))
            val_loss = loss_fn(val_pred, torch.from_numpy(Y_test).float()).item()
        progress_bar.update(1)
        epoch_loss /= len(batches)
        running_loss.append(epoch_loss)
        running_val_loss.append(val_loss)
        print("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, epoch_loss, val_loss))

    return network, running_loss, running_val_loss

if __name__ == "__main__":
    np.random.seed(43)
    torch.manual_seed(43)
    embedding_size = 8
    window = 5
    with open('LargerCorpus.txt', 'r', encoding='UTF-8') as f:
        txt = f.read()
    Filt_lemmas, w2i, i2w = prepare_texts(txt)
    network, running_loss, running_val_loss = train_sgns(txt, window, embedding_size)
    # Plot the training and validation losses
    plt.figure()
    plt.plot(running_loss, label="Training Loss")
    plt.plot(running_val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
    # save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/loss.png')
    # save model
    torch.save(network.state_dict(), 'sgns.pth')
    # # save embedding
    # embedding = network.embeddings.weight.data.numpy()
    # np.save('embedding.npy', embedding)
    # # visualize embedding

