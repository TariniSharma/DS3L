import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from load_dataset import *
import copy
import transform
from wideresnet import WideResNet, CNN, WNet

import argparse
import math
import time

import os

import imgaug.augmenters as iaa

# def get_augmenter():
#     seq = iaa.Sequential([
#         iaa.Crop(px=(0, 16)),
#         iaa.Fliplr(0.5),
#         iaa.GaussianBlur(sigma=(0, 3.0))
#     ])
#     def augment(images):
#         # print(images.shape)
#         return seq.augment(images = images.cpu().numpy())
#     return augment

# def sharpen(x, T):
#     temp = x**(1/T)
#     return temp / temp.sum(axis=1, keepdims=True)

# def mixup(x1, x2, y1, y2, alpha):
#     beta = torch.distributions.Beta(alpha, 1-alpha).sample()
#     print(beta)
#     x = torch.mul(x1, beta) + torch.mul(x2, (1 - beta))
#     y = torch.mul(y1, beta) + torch.mul((1 - beta), y2)
#     return x, y


# def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
#     # torch.to('cuda')
#     print("Y_SHAPE: ", y.shape)
#     xb = torch.tensor(augment_fn(x)).to('cuda')
#     print("xb_SHAPE: ", xb.shape)
#     ub = [torch.tensor(augment_fn(u)).to('cuda') for _ in range(K)]
#     print("ub_len_SHAPE: ", len(ub))
#     print("Y_SHAPE: ",y.shape)
#     print("model_SHAPE: ",model(ub[0]).shape)
#     qb = sharpen(sum(map(lambda i: model(i), ub)) / K, T)
#     print("qb_SHAPE: ", qb.shape)
#     QB = torch.tensor([qb[i].argmax() for i in range(qb.shape[0])]).to('cuda') 
#     # print(y.unique())
#     # print(qb)
#     print(QB.shape)
#     Ux = torch.cat(ub, axis = 0)
#     print("ux_SHAPE: ", Ux.shape)
#     Uy = torch.cat([QB for _ in range(K)], axis=0)
#     print("UY_SHAPE: ", Uy.shape)
#     t = torch.arange(len(xb) + len(Ux))
#     indices = t[torch.randperm(t.shape[0])]
#     Wx = torch.cat([Ux, xb], axis=0)[indices]
#     Wy = torch.cat([QB, y], axis=0)[indices]
#     print(Wx.shape, Wy.shape)
#     print(Ux.shape, Uy.shape)
#     X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
#     U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
#     return X, U, p, q


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    def augment(images):
        return seq.augment(images=images)
    return augment
    
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)
     

def mixup(x1, x2, alpha):
    # print(x1.shape, x2.shape)
    # print(y1.shape, y2.shape)
    beta = np.random.beta(alpha, alpha)
    beta = max(beta, 1 - beta)
    # print(x1.shape, x2.shape)
    x = beta * x1 + (1 - beta) * x2
    return x

# def find_labels(p, y_):
#   # print("HEY")
#   y = np.zeros(p.shape, dtype = 'int')
#   for i in range(p.shape[0]):
#     dist = math.inf
#     cls = 0
#     for k in y_.keys():
#       d = abs(y_[k]-p[i])**2
#       if(d <= dist):
#         dist = d
#         cls = k
#         # break
#     p[i] = cls
#   return y


def mixmatch(x, y, u, u__, model, augment_fn, T=0.5, K=2, alpha=0.75):
    # print(y.shape)
    y_ = np.zeros(y.shape, dtype ='int')
    for i in range(y_.shape[0]):
      y_[i] = y[i]
    # print(u.shape)
    ub = [augment_fn(u) for _ in range(K)] 
    ul = np.concatenate([u__ for _ in range(K)], axis=0) #unlabeled 
    u_hat = np.concatenate(ub, axis=0) # unlabeled and augmented
    xb = augment_fn(x) #augmented labeled set
    x____ = np.concatenate([u_hat, xb], axis =0) #150
    y____ = np.concatenate([ul, y], axis = 0) # 150
    indices = np.arange(x____.shape[0])
    # np.random.shuffle(indices)
    x____ = x____[indices]
    y____ = y____[indices]
    x_x = mixup(xb, x____[:y.shape[0]], alpha)
    u_u = mixup(u_hat, x____[y.shape[0]:], alpha)
    # print(x_x.shape)
    # print(u_u.shape)
    l_images, u_images = x_x, u_u
    l_labels, u_labels = y____[:y.shape[0]], y____[y.shape[0]:]
    # print(l_labels.shape, u_labels.shape)
    return torch.tensor(x).to('cuda'), torch.tensor(u).to('cuda'), torch.tensor(y).to('cuda'), torch.tensor(u__).to('cuda')
    # unique, frequency = np.unique(y_, return_counts = True) 
    # # y_mp = {}
    # y = np.zeros((len(y_), len(unique)))
    # # for i in range(len(unique)):
    # #   y_mp[unique[i]] = frequency[i]/sum(frequency)
    # # for i in range(y.shape[0]):
    # #   for c in range(len(unique)):
    # #     y[i][c] = y_mp[y_[i]] * (y_mp[y_[i]]*y_mp[c])
    # # y = np.array([y_mp[i]/frequency[i] for i in y_])
    # qb = sharpen(sum(map(lambda i: model(torch.tensor(i).to('cuda')), ub)) / K, T)
    # QB = qb.cpu().detach().numpy()
    # print(QB.shape)
    # # QB = np.array([i.argmax() for i in QB])
    # # QB = np.concatenate((QB, QB), axis = 0)
    # # print(QB.shape)
    # Ux = np.concatenate(ub, axis = 0)
    # Uy = np.concatenate([QB for _ in range(K)], axis=0)
    # # print(len(xb) + len(Ux))
    # indices = np.arange(len(xb) + len(Ux))
    # np.random.shuffle(indices)
    # Wx = np.concatenate([Ux, xb], axis=0)[indices]
    # Wy = np.concatenate([Uy, y], axis=0)[indices]
    # # print(Uy.shape, Ux.shape, Wx.shape, Wy.shape)
    # # print(np.random.shuffle(np.arange(len(xb) + len(Ux))))
    # X= mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    # U= mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    # # def compute_labels(y__):
    # #   return [y__[i].argmax() for i in range(y__.shape[0])]
    # print("XL", xl.shape)
    # y_hat = np.concatenate([xl, y_], axis=0)[indices]
    # py = y_hat[:len(xb)]
    # # # print(py)
    # qy = y_hat[len(xb):]
    # # p, q = compute_labels(p), compute_labels(q)
    # # p = find_labels(p, y_mp)
    # # q = find_labels(q, y_mp)
    # print(X.shape)
    # print(U.shape)
    # # print(py.shape)
    # # print(qy.shape)
    # # del p, q, y_hat, xb
    # return torch.tensor(X).to('cuda'), torch.tensor(U).to('cuda'), torch.tensor(py).to('cuda'), torch.tensor(qy).to('cuda')




parser = argparse.ArgumentParser(description='manual to this script')

#model
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--width', type=int, default=2)


#optimization
parser.add_argument('--optim', default='adam')
parser.add_argument('--iterations', type=int, default=200000)
parser.add_argument('--l_batch_size', type=int, default=100)
parser.add_argument('--ul_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--lr_decay_iter', type=int, default=400000)
parser.add_argument('--lr_decay_factor', type=float, default=0.2)
parser.add_argument('--warmup', type=int, default=200000)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--lr_wnet', type=float, default=6e-5) # this parameter need to be carefully tuned for different settings

#dataset
parser.add_argument('--dataset', default='MNIST')
parser.add_argument('--n_labels', type=int, default=60)
parser.add_argument('--n_unlabels', type=int, default=20000)
parser.add_argument('--n_valid', type=int, default=5000)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--tot_class', type=int, default=10)
parser.add_argument('--ratio', type=float, default=0.6)


args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benckmark = True
else:
    device = "cpu"

class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, model, mask):
        y_hat = model(x)
        return (F.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction='none').mean(1)*mask)

def build_model():

    if(args.dataset == 'CIFAR10'):
        transform_fn = transform.transform()
        model = WideResNet(widen_factor=args.width, n_classes=args.n_class, transform_fn=transform_fn).to(device)
    if(args.dataset == 'MNIST'):
        model = CNN(n_out=args.n_class).to(device)
    return model

def bi_train(model, label_loader, unlabeled_loader, val_loader, test_loader, optimizer, ssl_obj):
    wnet = WNet(6, 100, 1).to(device)

    wnet.train()

    t = time.time()
    best_acc = 0.0
    test_acc = 0.0
    iteration = 0

    optimizer_wnet = torch.optim.Adam(wnet.params(), lr=args.lr_wnet)

    for l_data, u_data in zip(label_loader, unlabeled_loader):

        #load data
        iteration += 1
        l_images, l_labels, _ = l_data
        u_images, u_labels, idx = u_data
        # print(type(l_images))
        if args.dataset == 'MNIST':
            l_images = l_images.unsqueeze(1)
            u_images = u_images.unsqueeze(1)
        l_images, l_labels = l_images.to(device).float(), l_labels.to(device).long()
        u_images, u_labels = u_images.to(device).float(), u_labels.to(device).long()

        model.train()
        meta_net = build_model()
        meta_net.load_state_dict(model.state_dict())
        # print(type(l_labels))
        # print(l_labels.shape)
        l_images, u_images, l_labels, u_labels = mixmatch(l_images.cpu().numpy(), l_labels.cpu().numpy(), u_images.cpu().numpy(), u_labels.cpu().numpy(), CNN(n_out=args.n_class).to(device),get_augmenter())
        # print(type(l_labels))
        # print(type(l_images))
        # print(type(l_labels))
        # cat labeled and unlabeled data
        labels = torch.cat([l_labels, u_labels], 0)
        labels[-len(u_labels):] = -1 #unlabeled mask
        unlabeled_mask = (labels == -1).float()
        images = torch.cat([l_images, u_images], 0)

        #coefficient for unsupervised loss
        coef = 10.0 * math.exp(-5 * (1 - min(iteration / args.warmup, 1)) ** 2)

        out = meta_net(images)
        ssl_loss = ssl_obj(images, out.detach(), meta_net, unlabeled_mask)

        cost_w = torch.reshape(ssl_loss[len(l_labels):], (len(ssl_loss[len(l_labels):]), 1))


        weight = wnet(out.softmax(1)[len(l_labels):])
        norm = torch.sum(weight)

        cls_loss = F.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
        if norm != 0:
            loss_hat = cls_loss + coef * (torch.sum(cost_w * weight) / norm + ssl_loss[:len(l_labels)].mean())
        else:
            loss_hat = cls_loss + coef * (torch.sum(cost_w * weight) + ssl_loss[:len(l_labels)].mean())
        # meta_net.to(device)
        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_hat, (meta_net.params()), create_graph=True)
        meta_net.update_params(lr_inner=args.meta_lr, source_params=grads)
        del grads

        #compute upper level objective
        y_g_hat = meta_net(l_images)
        l_g_meta = F.cross_entropy(y_g_hat, l_labels)

        optimizer_wnet.zero_grad()
        l_g_meta.backward()
        optimizer_wnet.step()

        out = model(images)

        ssl_loss = ssl_obj(images, out.detach(), model, unlabeled_mask)
        cls_loss = F.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
        cost_w = torch.reshape(ssl_loss[len(l_labels):], (len(ssl_loss[len(l_labels):]), 1))
        with torch.no_grad():
            weight = wnet(out.softmax(1)[len(l_labels):])
            norm = torch.sum(weight)

        if norm != 0:
            loss = cls_loss + coef * (torch.sum(cost_w * weight) / norm + ssl_loss[:len(l_labels)].mean())
        else:
            loss = cls_loss + coef * (torch.sum(cost_w * weight) + ssl_loss[:len(l_labels)].mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration == 1 or (iteration % 1000) == 0:
            time_cost = time.time() - t
            print("iteration [{}/{}] cls loss : {:.6e},  time : {:.3f} sec/iter, lr : {}, coef: {}".format(
                iteration, args.iterations, loss.item(),  time_cost / 100, optimizer.param_groups[0]["lr"], coef))
            t = time.time()

        if (iteration % 10000) == 0 or iteration == args.iterations:
            acc = test(model, val_loader)
            print("Validation Accuracy: {}".format(acc))
            if (acc > best_acc):
                best_acc = acc
                test_acc = test(model, test_loader)
            model.train()
        if iteration == args.lr_decay_iter:
            optimizer.param_groups[0]['lr'] *= args.lr_decay_factor
    print("Last Model Accuracy: {}".format(test(model, test_loader)))
    print("Test Accuracy: {}".format(test_acc))

args.meta_lr = 0.005
def test(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        tot = 0.
        for i, data in enumerate(test_loader):
            images, labels, _ = data

            if args.dataset == 'MNIST':
                images = images.unsqueeze(1)

            images = images.to(device).float()
            labels = labels.to(device).long()

            out = model(images)

            pred_label = out.max(1)[1]
            correct += (pred_label == labels).float().sum()
            tot += pred_label.size(0)
        acc = correct / tot
        return acc

def main():

    args.l_batch_size = args.l_batch_size // 2
    args.ul_batch_size = args.ul_batch_size // 2

    data_loaders = get_dataloaders(dataset=args.dataset, n_labels=args.n_labels, n_unlabels=args.n_unlabels, n_valid=args.n_valid,
                                   l_batch_size=args.l_batch_size, ul_batch_size=args.ul_batch_size,
                                   test_batch_size=args.test_batch_size, iterations=args.iterations,
                                   tot_class=args.n_class, ratio=args.ratio)
    label_loader = data_loaders['labeled']
    unlabel_loader = data_loaders['unlabeled']
    test_loader = data_loaders['test']
    val_loader = data_loaders['valid']


    model = build_model()

    if(args.dataset=="MNIST"):
        optimizer = torch.optim.SGD(model.params(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.params(), lr=3e-4)


    U_Loss = MSE_Loss()
    bi_train(model, label_loader, unlabel_loader, val_loader, test_loader, optimizer, U_Loss)

if __name__ == '__main__':
    main()
