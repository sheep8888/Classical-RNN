import torch
from model.rnn import RNNModel
from model.gru import GRUModel
from model.lstm import LSTMModel
from data import load_data_time_machine


import argparse 


parser = argparse.ArgumentParser(description='Parameters') 

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_hiddens', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--use_random_iter', action='store_true')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--token', type=str, choices=['char','word'] ,default='word')
parser.add_argument('--model_name', type=str, choices=['RNN', 'GRU', 'LSTM'] ,default='RNN')
args = parser.parse_args()

num_epochs, lr = args.epochs,args.learning_rate 
num_hiddens = args.num_hiddens
batch_size = args.batch_size
num_steps = args.num_step
use_random_iter= args.use_random_iter
token = args.token
num_layers = args.num_layers

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter, token=token)
    
if args.model_name == 'RNN':
    net = RNNModel(len(vocab), num_hiddens, num_layers)
elif args.model_name == 'GRU':
    net = GRUModel(len(vocab), num_hiddens, num_layers)
else:
    net = LSTMModel(len(vocab), num_hiddens, num_layers)
train(net, train_iter, vocab, lr, num_epochs, device=device, use_random_iter=use_random_iter, token=token)
