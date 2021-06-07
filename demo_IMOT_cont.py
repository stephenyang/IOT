
import numpy as np
import matplotlib.pyplot as plt
import ot
np.random.seed(123456)

import matplotlib
matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)

epsilon = 0.1
batch_size = 8000

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
from torch.autograd import Variable

parser = argparse.ArgumentParser()
# parser.add_argument('--inputsize', type=int, default=1, help='the height / width of the input to network')
# parser.add_argument('--nc', type=int, default=1, help='input size')
# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--ngpu'  , type=int, default=0, help='number of GPUs to use')
# parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
# parser.add_argument('--lrG', type=float, default=0.1, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

opt = parser.parse_args()
print(opt)
# ngpu = int(opt.ngpu)
# nz = int(opt.nz)
# ngf = int(opt.ngf)
# ndf = int(opt.ndf)
# nc = int(opt.nc)
# n_extra_layers = int(opt.n_extra_layers)

n = 100
x = np.linspace(0.01, 1, num=n)

p1 = 0.55 * ot.datasets.get_1D_gauss(n,20,18) + 0.45 * ot.datasets.get_1D_gauss(n,80,9)
p2 = 0.55 * ot.datasets.get_1D_gauss(n,15,9) + 0.45 * ot.datasets.get_1D_gauss(n,65,15)

plt.plot(x,p1,'o-',color='blue')
plt.plot(x,p2,'o-',color='red')
plt.tight_layout()
plt.show()

psi = 2
c = np.absolute(psi*x-np.expand_dims(x,axis=1))
# c = c/n
# print(c)
p = 2
rho = 1
c = np.power(c,p)
print(c)
plt.imshow(c)
plt.colorbar()
plt.clim(0,1)
plt.show()

# gamma = 1
K = np.exp(-c/epsilon)
u = np.ones([n,])
v = u
maxiter = 100
for iter in range(maxiter):
    u = p1/np.matmul(K,v)
    v = p2/np.matmul(np.transpose(K),u)
pi = np.expand_dims(u,axis=1)*K*np.expand_dims(v,axis = 0)
plt.imshow(pi)
plt.show()

plt.plot(np.sum(pi,axis = 0))
plt.plot(np.sum(pi,axis = 1))
plt.show()
pivec = np.array(pi).flatten()#pi.reshape([1,n*n])

nsamps = 1100000
samp = np.zeros((nsamps,2))
seq = np.linspace(0, n*n-1, num=n*n)
numsamp = (pivec*nsamps).astype(int)
print(np.sum(numsamp))
i = 0
for idx in range(len(numsamp)):
    ct = numsamp[idx]
    rowidx = np.int_(idx / n)#.astype(int)
    colidx = (np.mod(idx,n)).astype(int)
    samp[i:(i+ct),] = [x[rowidx],x[colidx]]
    i+=ct

#### IOT


# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        nh = 20
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1,nh),
            nn.Tanh(),
            nn.Linear(nh,nh),
            nn.Tanh(),
            nn.Linear(nh,nh),
            nn.Tanh(),
            nn.Linear(nh,nh),
            nn.Tanh(),
            nn.Linear(nh,1),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.ndim != 0:
            x = x.view(x.shape[0],1)
        else:
            x = x.view(1,1)
        output = self.linear_relu_stack(x)
        output = output.mean(0)
        return output.view(1)

netDalpha = NeuralNetwork().to(device)# mlp.MLP_D(opt.inputsize, nz, nc, ndf, ngpu)
# netDalpha.apply(weights_init)

netDbeta = NeuralNetwork().to(device)#mlp.MLP_D(opt.inputsize, nz, nc, ndf, ngpu)
# netDbeta.apply(weights_init)

netDeta = NeuralNetwork().to(device)#mlp.MLP_D(opt.inputsize, nz, nc, ndf, ngpu)
# netDeta.apply(weights_init)

# one = torch.FloatTensor([1])
# mone = one * -1

# setup optimizer
params_alpha = list(netDalpha.parameters())
params_beta = list(netDbeta.parameters())
params_eta = list(netDeta.parameters())
if opt.adam:
    optimizerAlpha = optim.Adam(params_alpha, lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerBeta = optim.Adam(params_beta, lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerEta = optim.Adam(params_eta, lr=opt.lrD, betas=(opt.beta1, 0.999))
else:
    optimizerAlpha = optim.RMSprop(params_alpha, lr=opt.lrD)
    optimizerBeta = optim.RMSprop(params_beta, lr=opt.lrD)
    optimizerEta = optim.RMSprop(params_eta, lr=opt.lrD)


x_train = samp[:,0]
y_train = samp[:,1]
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = x_train.float()
y_train = y_train.float()
train_dataset = TensorDataset(x_train,y_train)

def customized_loss(x, y):
    return -netDalpha(x_batch)-netDbeta(y_batch)+netDeta(torch.pow(torch.abs(psi*x-y),rho))+epsilon*torch.exp((netDalpha(x_batch)+netDbeta(y_batch)-netDeta(torch.pow(torch.abs(psi*x-y),rho)))/epsilon)
# print(train_dataset.size())
# fullsamp_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
xfull= torch.Tensor(x_train)
yfull= torch.Tensor(y_train)
# xloader = torch.utils.data.DataLoader(dataset=xfull, batch_size=1, shuffle=False).to(device)
# yloader = torch.utils.data.DataLoader(dataset=yfull, batch_size=1, shuffle=False).to(device)
epochs = 1
# psi = torch.Tensor(psi)
for t in range(epochs):
    samp_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for batch, (x_batch, y_batch) in enumerate(samp_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = customized_loss(x_batch,y_batch)

        netDalpha.zero_grad()
        netDbeta.zero_grad()
        netDeta.zero_grad()

        loss.backward()

        optimizerAlpha.step()
        optimizerBeta.step()
        optimizerEta.step()

        # loss = customized_loss(xfull, yfull)
        if batch%100==0:
            print('[%d] %f'%(batch,loss))

netDeta.eval()
cost = torch.zeros((n,n))
# alpha = torch.zeros(n)
# beta = torch.zeros(n)
# cost_input = np.zeros((n,n))
for x in range(n):
    for y in range(n):
        # xinput = torch.ones((20,1))*x/n
        # yinput = torch.ones((20,1))*y/n
        # cost_input[y,x] = np.power(np.abs(psi*((x+1)/n)-(y+1)/n),1)
        test = np.power(np.abs(psi*(x/n)-y/n),rho)
        test = torch.from_numpy(np.array(test))
        test = test.float()
        cost[y,x]=netDeta(torch.abs(test))
        # alpha[x] = netDalpha(xinput)
        # beta[y] = netDbeta(yinput)
#cost = cost/torch.max(cost)
plt.imshow(cost.detach().numpy())
plt.colorbar()
# plt.clim(0,1)
plt.show()
print(cost)
# exp_alpha = torch.exp(alpha)
#
# diag_u = torch.diag(exp_alpha)
#
# exp_beta = torch.exp(beta)
# diag_v = torch.diag(exp_beta)

# pi_est = diag_u * torch.exp(-cost/epsilon) * diag_v
# plt.imshow(pi_est.detach().numpy())
# plt.show()
# print(alpha)
# print(beta)
# print(pi_est)

# u = exp_alpha.detach().numpy()
# v = exp_beta.detach().numpy()
# K = torch.exp(-cost/epsilon).detach().numpy()
# pi = np.expand_dims(u,axis=1)*K*np.expand_dims(v,axis = 0)
# plt.imshow(pi)
# plt.show()

# x = np.linspace(0.01, 1, num=n)
# print(np.absolute(psi*x-np.expand_dims(x,axis=1)))
# print(cost_input)