import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DNNModel(nn.Module):
    def __init__(self, ninp, nhid, nout, dropout=0.5):
        super(DNNModel, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(nhid, nout),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

