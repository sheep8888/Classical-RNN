import torch
from torch import nn
from torch.nn import functional as F


class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens=512, num_layers=1, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.rnn = nn.RNN(vocab_size, num_hiddens, num_layers)
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        
    def forward(self, X, state):
        # 这里简单地使用one hot编码（也可使用更好的词嵌入）
        X = F.one_hot(X.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    
    def init_state(self, batch_size, device):
        return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),device=device)
