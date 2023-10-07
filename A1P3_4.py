import torch


class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        # initialize weight matrix to random numbers
        self.linear = torch.nn.Linear(embedding_size, vocab_size)
        self.linear.weight.data.uniform_(-0.5, 0.5)
        self.linear.bias.data.zero_()

        # prediction function takes embedding as input, and predicts which word in vocabulary as output

    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        e = self.embeddings(x)
        logits = self.linear(e)
        # logits: torch.tensor of shape (bsz, vocab_size

        return logits, e
