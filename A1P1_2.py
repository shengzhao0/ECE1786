import torch
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)


def print_closest_cosine_words(vec, n=5):
    dists_cos = torch.cosine_similarity(
        glove.vectors, vec.unsqueeze(0))
    lst_cos = sorted(enumerate(dists_cos.numpy()),
                     key=lambda x: x[1], reverse=True)
    print("Cosine similarity")
    for idx, difference in lst_cos[1:n+1]:
        print(glove.itos[idx], "\t%5.2f" % difference)


def print_closest_words(vec, n=5):
    # compute distances to all words
    dists = torch.norm(glove.vectors - vec, dim=1)
    lst = sorted(enumerate(dists.numpy()),
                 key=lambda x: x[1])  # sort by distance
    print("Euclidean distance")
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)


print_closest_cosine_words(glove["dog"], n=10)
print_closest_words(glove['dog'], 10)

print_closest_cosine_words(glove['computer'], 10)
print_closest_words(glove['computer'], 10)
