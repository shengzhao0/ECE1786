import torch


class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_u = torch.nn.Embedding(vocab_size, embedding_size)
        self.embeddings_v = torch.nn.Embedding(vocab_size, embedding_size)
        #self.logsigmoid = torch.nn.LogSigmoid()
        #self.embeddings.weight.data.uniform_(-0.1, 0.1)
        # initialize weight matrix to random numbers

        
        # TO DO
        
    def forward(self, x, t):
        
        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.
        x = self.embeddings_u(x)
        t = self.embeddings_v(t)
        # DO THE DOT PRODUCT OF THE TWO EMBEDDINGS AND RETURN THE LOGITS
        logits = torch.bmm(x.unsqueeze(1), t.unsqueeze(2)).squeeze()


        # TO DO

        return logits