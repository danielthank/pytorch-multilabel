import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DNNModel(nn.Module):
    def __init__(self, ninp, nhid, nout, dropout=0.5):
        super(DNNModel, self).__init__()

        self.lin = nn.Linear(ninp, nhid)
        self.l1 = nn.Linear(nhid, nhid)
        self.l2 = nn.Linear(nhid, nhid)
        self.lout = nn.Linear(nhid, nout)
        self.lth = nn.Linear(nhid, nhid)
        self.lthout = nn.Linear(nhid, 1)

        self.drop = dropout

    def forward(self, x):
        tmp = F.dropout(F.relu(self.lin(x)), self.drop)
        out = F.dropout(F.relu(self.l1(tmp)), self.drop)
        out = F.dropout(F.relu(self.l2(out)), self.drop)
        out = F.sigmoid(self.lout(tmp))

        tmp = F.dropout(F.relu(self.lth(tmp)), self.drop)
        threshold = F.sigmoid(self.lthout(tmp))

        return out, threshold

