import torch
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)


def compare_words_to_category(cat, vec):
    sim_lst = []
    cat_lst = []
    for i in cat:
        cat_lst.append(glove[i].unsqueeze(0))
    for word in cat_lst:
        dis = torch.cosine_similarity(
            word, vec.unsqueeze(0))
        sim_lst.append(dis)
    a1 = sum(sim_lst) / len(sim_lst)
    mean_embedding = torch.mean(torch.cat(cat_lst, dim=0), 0)
    a2 = torch.cosine_similarity(mean_embedding.unsqueeze(0), vec.unsqueeze(0))
    return a1, a2


if __name__ == '__main__':
    Cat = ['sun', 'moon', 'winter', 'rain', 'cow', 'wrist',
           'wind', 'jprefix', 'ghost', 'glow', 'heated', 'cool']
    vec = glove['apple']
    a1, a2 = compare_words_to_category(Cat, vec)
    print("apple")
    print("a1: ", a1, "\ta2: ", a2)
