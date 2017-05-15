import argparse
import torch
import torch.nn as nn
import time
from torch import optim
from torch.autograd import Variable

import read_data
import model

parser = argparse.ArgumentParser(description='PyTorch HW5')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=32,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='save/model.pt',
                    help='path to save the final model')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(nbatch, bsz, -1)
    if args.cuda:
        data = data.cuda()
    return data

corpus = read_data.Corpus('data/')
ninp = corpus.train.size(1)
nout = corpus.train_targets.size(1)

train_data = batchify(corpus.train, args.batch_size)
print('train_data', train_data.size())
train_targets = batchify(corpus.train_targets, args.batch_size)
print('train_targets', train_targets.size())
val_data = batchify(corpus.val, args.batch_size)
print('val_data', val_data.size())
val_targets = batchify(corpus.val_targets, args.batch_size)
print('val_targets', val_targets.size())
test_data = batchify(corpus.test, args.batch_size)
print('test_data', test_data.size())

lr = args.lr
best_val_loss = None
dnn = model.DNNModel(ninp, args.nhid, nout, args.nlayer, dropout=args.dropout)
if args.cuda:
    dnn.cuda()
optimizer = optim.Adam(dnn.parameters())
criterion = nn.MultiLabelSoftMarginLoss()

def train():
    dnn.train()
    total_loss = 0
    start_time = time.time()
    for i, (data_batch, target_batch) in enumerate(zip(train_data, train_targets)):
        data_batch = Variable(data_batch)
        target_batch = Variable(target_batch)
        
        output = dnn(data_batch)
        loss = criterion(output, target_batch)

        dnn.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, i, len(train_data), lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate():
    dnn.eval()
    total_loss = 0
    for data_batch, target_batch in zip(val_data, val_targets):
        data_batch = Variable(data_batch)
        target_batch = Variable(target_batch)

        output = dnn(data_batch)
        total_loss += criterion(output, target_batch).data

    return total_loss[0] / val_data.size(0)

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'
              .format(epoch, time.time() - epoch_start_time, val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(dnn, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    dnn = torch.load(f)
