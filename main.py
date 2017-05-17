import argparse
import torch
import torch.nn as nn
import time
import numpy as np
import sys
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.metrics import f1_score

import read_data
import model

parser = argparse.ArgumentParser(description='PyTorch HW5')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=1024,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--test', action='store_true',
                    help='test')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
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

corpus = read_data.Corpus('data/')
ninp = corpus.train_data.size(1)
nout = corpus.train_targets.size(1)

td = corpus.train_data.cuda()
tt = corpus.train_targets.cuda()
vd = corpus.val_data.cuda()
vt = corpus.val_targets.cuda()
test = corpus.test_data.cuda()

if args.test:
    with open(args.save, 'rb') as f:
        dnn = torch.load(f)

    with open('out.csv', 'w') as f:
        print('"id","tags"', file=f)
        dnn.eval()
        z = 0
        for i in range(0, test.size(0), args.batch_size):
            data_batch = Variable(test[i:i+args.batch_size])
            output = dnn(data_batch)
            predict = torch.gt(output, 0.5).data.cpu().numpy().astype('int32')
            for j in range(predict.shape[0]):
                if np.all(predict[j]==0):
                    predict[j, np.argmax(output[j])] = 1
                line = []
                for k in range(predict.shape[1]):
                    if predict[j][k] == 1:
                        line.append(corpus.tags[k])
                print('"{}","{}"'.format(z, ' '.join(line)), file=f)
                z = z+1
    sys.exit(0)

lr = args.lr
best_val_loss = None
dnn = model.DNNModel(ninp, args.nhid, nout, dropout=args.dropout)
if args.cuda:
    dnn.cuda()
optimizer = optim.Adam(dnn.parameters())
criterion = nn.BCELoss()

def f1_batch(pred, ground):
    f1 = 0
    for i in range(pred.shape[0]):
        f1 = f1 + f1_score(ground[i], pred[i])
    return f1

def train():
    dnn.train()
    total_loss = 0
    start_time = time.time()
    z = 0
    baseline = 0
    for i in range(0, td.size(0), args.batch_size):
        z = z+1
        data_batch = Variable(td[i:i+args.batch_size])
        target_batch = Variable(tt[i:i+args.batch_size])
        
        output = dnn(data_batch)
        loss = criterion(output, target_batch)
        total_loss += loss.data
        dnn.zero_grad()
        loss.backward(retain_variables=True)

        action = output.bernoulli()
        predict = action.data.cpu().numpy().astype('int32')
        ground = target_batch.data.cpu().numpy().astype('int32')
        reward = f1_batch(predict, ground) / predict.shape[0]
        baseline = baseline * 0.9 +  reward * 0.1
        action.reinforce(reward - baseline)
        autograd.backward([action], [None])

        optimizer.step()
        if z % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | loss {:5.2f}'
                  .format(epoch, z, td.size(0) // args.batch_size, lr, elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate():
    dnn.eval()
    total_loss = 0
    f1 = 0
    for i in range(0, vd.size(0), args.batch_size):
        data_batch = Variable(vd[i:i+args.batch_size])
        target_batch = Variable(vt[i:i+args.batch_size])

        output = dnn(data_batch)
        total_loss += criterion(output, target_batch).data

        predict = torch.gt(output, 0.5).data.cpu().numpy().astype('int32')
        ground = target_batch.data.cpu().numpy().astype('int32')
        for j in range(predict.shape[0]):
            if np.all(predict[j]==0):
                predict[j, np.argmax(output[j])] = 1

        f1 = f1 + f1_batch(predict, ground)

    return total_loss[0] / vd.size(0), f1 / vd.size(0)

best_f1_score = None
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, f1 = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | f1 score {:5.3f}'
              .format(epoch, time.time() - epoch_start_time, val_loss, f1))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_f1_score or f1 > best_f1_score:
            with open(args.save, 'wb') as f:
                torch.save(dnn, f)
            best_f1_score = f1

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

