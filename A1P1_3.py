import torch
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)
first_word = ['small', "large", "easy", "high",
              "low", "slow", "tall", "short", "fat", "thin"]


def print_closest_sec_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)
    lst = sorted(enumerate(dists.numpy()),
                 key=lambda x: x[1])  # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)


for i in first_word:
    print("Second word of",i, ":")
    print_closest_sec_words(glove['greater']-glove['great']+glove[i])
