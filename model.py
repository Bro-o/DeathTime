from torch import nn
from DeathTime.tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.emb = nn.Embedding()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.drop = nn.Dropout(emb_dropout)

    def init_weights(self):
        initrange = 0.1

    def forward(self, x):
        emb = self.drop(self.emb(x))
        y = self.tcn() # input should have dimension (N, C, L)