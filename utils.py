import time
from tqdm import tqdm
import torch
from torch import nn
import math


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


  def predict(prefix, num_preds, net, vocab, device, token='word'):
    """在prefix后面生成新单词"""
    net.eval()
    if token == 'word':
        prefix = prefix.split(' ')
    with torch.no_grad():
        state = net.init_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
        for y in prefix[1:]:  # 预热期
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):  # 预测num_preds步
            y, state = net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        if token == 'word':
            return ' '.join([vocab.idx_to_token[i] for i in outputs])
        elif token == 'char':
            return ''.join([vocab.idx_to_token[i] for i in outputs])
        else:
            print('错误：未知词元类型：' + token)


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):

    start_t = time.time()
    state = None
    batchs, words, train_val_loss = 0, 0, 0.0
    for X, Y in tqdm(train_iter):
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.init_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        batchs = batchs + 1
        words = word + len(y)
        train_val_loss = train_val_loss + l.item()
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

    end_t = time.time()
    return math.exp(train_val_loss/batchs), words/(end_t-start_t)


def train(net, train_iter, vocab, lr, num_epochs=500, device='cpu', use_random_iter=False, print_predict=False, token="word"):
    net.to(device)
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    predict_ = lambda prefix: predict(prefix, 10, net, vocab, device, token)
    # 训练和预测
    for epoch in range(num_epochs):
        net.train()
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        print(f'epoch {epoch+1} 困惑度 {ppl:.2f}')
        if print_predict and (epoch + 1) % 10 == 0:
            predict_('time traveller')
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 批次/秒 {str(device)}')
    print(predict_('time traveller'))
    print(predict_('traveller'))
