from PIL import Image

import torch
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib.pyplot import figure


import random
import scipy

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from numpy import *
from torch import Tensor
from torch.nn import Parameter
from random import sample


def get_samples(num, flag):
    cat_tensor = torch.zeros(num, 3)
    #     Samples = torch.zeros(num, 3)
    if flag == 1:
        list1 = list(expsamples1)
        random_list1 = sample(list1, num)
        for i in range(num):
            cat_tensor[i] = random_list1[i]
    elif flag == 2:
        list2 = list(expsamples2)
        random_list2 = sample(list2, num)
        for i in range(num):
            cat_tensor[i] = random_list2[i]

    return cat_tensor


def loss_f(net1, net2, x, y, epsilon):
    cxy = (torch.norm(x - 2 * y, dim=1)).reshape(-1, 1)
    LF = (net1(x) + net2(y) - epsilon * (torch.exp((net1(x) + net2(y) - cxy ** 2) / epsilon))).mean()
    return LF


def loss_h(net1, net2, net3, x, y, epsilon):
    dxy = (torch.norm(2 * y - net3(x), dim=1)).reshape(-1, 1)
    cxy = (torch.norm(x - 2 * y, dim=1)).reshape(-1, 1)
    LH = ((torch.exp((net1(x) + net2(y) - cxy ** 2) / epsilon)) * (dxy ** 2)).mean()
    return LH


def loss_fn(net1, net2, net4, x, y, epsilon):
    #     cxy = (torch.norm(x-y, dim = 1)).reshape(-1, 1)
    xy = torch.abs(x - y)
    LF = (net1(x) + net2(y) - epsilon * (torch.exp((net1(x) + net2(y) - net4(xy)) / epsilon))).mean()
    return LF


def loss_hn(net1, net2, net3, net4, x, y, epsilon):
    dxy = (torch.norm(y - net3(x), dim=1)).reshape(-1, 1)
    xy = torch.abs(x - y)
    #     cxy = (torch.norm(x-y, dim = 1)).reshape(-1, 1)
    LH = ((torch.exp((net1(x) + net2(y) - net4(xy)) / epsilon)) * (dxy ** 2)).mean()
    return LH


class networku(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(networku, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


class networkv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(networkv, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)

        return output


class networkf(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(networkf, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        m.bias.data.fill_(0)

    elif type(m) == nn.BatchNorm1d:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def plot_samples_traj(out_iter):
    x_test = get_samples(im1.size[0] * im1.size[1], 1)
    # x_test = torch.Tensor(x_test)
    # x_test = autograd.Variable(x_test, requires_grad = True)

    test_output = net_f(x_test).data.numpy()
    cr = scale * test_output
    #     cr = np.abs(cr/np.amax(cr))
    cr = (cr - np.amin(cr)) / (np.amax(cr) - np.amin(cr))
    # color_array = test_output

    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    # plt.xlim([min_x, max_x])
    # plt.ylim([min_y, max_y])
    # plt.xticks(np.arange(min_x, max_x, xspace))
    # plt.yticks(np.arange(min_y, max_y, yspace))
    ax = plt.axes(projection='3d')
    ax.scatter3D(test_output[:, 0], test_output[:, 1], test_output[:, 2], c=cr, marker='.', s=10)
    #     plt.title('1 to 2 iteration = {}'.format(out_iter)+ '.jpg')
    plt.savefig('test/1 to 2 iteration = {}'.format(out_iter) + '.jpg')
    plt.close()


def color_transf(iteration):
    newsamples1 = scale * (net_f(expsamples1))

    im1 = Image.open('source_cp.tiff')  # Can be many different formats.
    pix1 = im1.load()

    Length1 = im1.size[0]
    Height1 = im1.size[1]

    for k in range(Length1):
        for l in range(Height1):
            p = newsamples1[k * Height1 + l].detach().numpy()
            pix1[k, l] = (max(p[0], 0), max(p[1], 0), max(p[2], 0))

    im1.save('test/color_transform sour to tar out iteration' + str(iteration) + '.jpg')

#####
##### learn u,v,pi with true c
im1 = Image.open('source_cp.tiff')  # Can be many different formats.
pix1 = im1.load()

Length1 = im1.size[0]
Height1 = im1.size[1]

samples1 = torch.zeros(Length1 * Height1, 3)

count = 0
dim = 3
scale = 50

for k in range(Length1):
    for l in range(Height1):
        samples1[count] = torch.tensor(pix1[k, l])
        count = count + 1

expsamples1 = samples1 / scale

samples11 = get_samples(Length1 * Height1, 1)

color_array1 = (samples11.numpy()) * scale

fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection='3d')
ax.scatter3D(samples11[:, 0], samples11[:, 1], samples11[:, 2], c=color_array1 / 255, marker='.', s=10);
plt.savefig('test/scatterplot of RGB values of picture 1')

plt.close()

im2 = Image.open('target_cp.tiff')  # Can be many different formats.
pix2 = im2.load()

Length2 = im2.size[0]
Height2 = im2.size[1]

samples2 = torch.zeros(Length2 * Height2, 3)

count = 0

for k in range(Length2):
    for l in range(Height2):
        samples2[count] = torch.tensor(pix2[k, l])
        count = count + 1

expsamples2 = samples2 / scale

samples22 = get_samples(Length1 * Height1, 2)

color_array2 = (samples22.numpy()) * scale

fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection='3d')
ax.scatter3D(samples22[:, 0], samples22[:, 1], samples22[:, 2], c=color_array2 / 255, marker='.', s=10);
plt.savefig('test/scatterplot of RGB values of picture 2')

plt.close()

net_u = networku(dim,36,1)
net_u.apply(weights_init)

net_v = networkv(dim,36,1)
net_v.apply(weights_init)



optimizer_u = optim.Adam(net_u.parameters(), lr = 1e-3)
optimizer_v = optim.Adam(net_v.parameters(), lr = 1e-3)



iter_uv = 20000
iter_f = 20000
batch_size = 2000
epsilon = 0.5

loss1_list = []

######## training process for dual variables
for i in range(iter_uv):
    x_sample = get_samples(batch_size, 1)
    x_sample = autograd.Variable(x_sample, requires_grad=True)
    y_sample = get_samples(batch_size, 2)
    y_sample = autograd.Variable(y_sample, requires_grad=True)

    for para in net_u.parameters():
        para.requires_grad = True
    for para in net_v.parameters():
        para.requires_grad = True

    net_u.zero_grad()
    net_v.zero_grad()

    LOSS1 = -loss_f(net_u, net_v, x_sample, y_sample, epsilon)
    LOSS1.backward()
    optimizer_u.step()
    optimizer_v.step()
    loss1_list.append(LOSS1.data.numpy())

    print('iter_uv:',i,-LOSS1.data.numpy())

net_f = networkf(dim, 36, dim)
net_f.apply(weights_init)
optimizer_f = optim.Adam(net_f.parameters(), lr=1e-3)

loss2_list = []

########## training process for optimal map
for j in range(iter_f):
    xx_sample = get_samples(batch_size, 1)
    xx_sample = autograd.Variable(xx_sample, requires_grad=True)
    yy_sample = get_samples(batch_size, 2)
    yy_sample = autograd.Variable(yy_sample, requires_grad=True)

    for para in net_u.parameters():
        para.requires_grad = False
    for para in net_v.parameters():
        para.requires_grad = False
    for para in net_f.parameters():
        para.requires_grad = True

    net_u.zero_grad()
    net_v.zero_grad()
    net_f.zero_grad()

    LOSS2 = loss_h(net_u, net_v, net_f, xx_sample, yy_sample, epsilon)
    LOSS2.backward()
    optimizer_f.step()
    loss2_list.append(LOSS2.data.numpy())

    print('iter:', j, 'w_loss:', LOSS2.data.numpy())

    ############### save figures
    if j % 200 == 0:
        plot_samples_traj(j)
        color_transf(j)

import random


def get_invsamples(num):
    #     cat_tensor1 = torch.zeros(num, 3)
    #     cat_tensor2 = torch.zeros(num, 3)
    #     list1 = list(invsamples1)
    #     list2 = list(invsamples2)
    #     random_list1 = sample(list1, num)
    indice = random.sample(range(Length1 * Height1), num)
    indice = torch.tensor(indice)
    cat_tensor1 = invsamples1[indice]
    cat_tensor2 = invsamples2[indice]
    #     for i in range(num):
    #         cat_tensor1[i] = random_list1[i]
    #         cat_tensor2[i] = random_list2[i]
    return cat_tensor1, cat_tensor2


def iot(neta, netb, netc, x, y):
    x1 = autograd.Variable(torch.Tensor(np.random.uniform(0, 1, [batch_2, 3])), requires_grad=True)
    y1 = autograd.Variable(torch.Tensor(np.random.uniform(0, 1, [batch_2, 3])), requires_grad=True)

    xy = torch.abs(x - y)
    xy1 = torch.abs(x1 - y1)
    l1 = neta(x).mean()
    l2 = netb(y).mean()
    l3 = netc(xy).mean()
    l4 = (iot_epsilon * (torch.exp((neta(x1) + netb(y1) - netc(xy1)) / iot_epsilon))).mean()
    lt = -l1 - l2 + l3 + l4
    return lt


######## inverse OT
### pick two figures
count = 0
dim = 3
scale = 255
#####################################
im1 = Image.open('source_cp.tiff')  # Can be many different formats.
pix1 = im1.load()
im2 = Image.open('source_iot.jpg')  # Can be many different formats.
pix2 = im2.load()
#####################################
Length1 = im1.size[0]
Height1 = im1.size[1]
#####################################
samples1 = torch.zeros(Length1 * Height1, 3)
samples2 = torch.zeros(Length1 * Height1, 3)

for k in range(Length1):
    for l in range(Height1):
        samples1[count] = torch.tensor(pix1[k, l])
        samples2[count] = torch.tensor(pix2[k, l])
        count = count + 1

invsamples1 = samples1 / scale
invsamples2 = samples2 / scale

samples11, samples22 = get_invsamples(Length1 * Height1)

color_array1 = samples11.numpy()
color_array2 = samples22.numpy()

fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection='3d')
ax.scatter3D(samples11[:, 0], samples11[:, 1], samples11[:, 2], c=color_array1, marker='.', s=10);
plt.savefig('inv scatterplot of RGB values of picture 1')
plt.close()

fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection='3d')
ax.scatter3D(samples22[:, 0], samples22[:, 1], samples22[:, 2], c=color_array2, marker='.', s=10);
plt.savefig('inv scatterplot of RGB values of picture 2')
plt.close()


# class Swish(nn.Module):
#     def forward(self, input_tensor):
#         return input_tensor * torch.sigmoid(input_tensor)

class networkc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(networkc, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            Swish(),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


######## define networks
iot_u = networku(dim, 36, 1)
iot_u.apply(weights_init)

iot_v = networkv(dim, 36, 1)
iot_v.apply(weights_init)

iot_c = networkc(dim, 36, 1)
iot_c.apply(weights_init)

optimizer_c = optim.Adam(iot_c.parameters(), lr=1e-3)
optimizer_iu = optim.Adam(iot_u.parameters(), lr=1e-3)
optimizer_iv = optim.Adam(iot_v.parameters(), lr=1e-3)
#########iot algorithm
iot_iter = 20000
batch_1 = 2000
batch_2 = 20000
iot_epsilon = 1
iotloss_list = []
diffl_list = []
diffm_list = []

for k in range(iot_iter):
    iotsamples1, iotsamples2 = get_invsamples(batch_1)
    iotsamples1 = autograd.Variable(iotsamples1, requires_grad=True)
    iotsamples2 = autograd.Variable(iotsamples2, requires_grad=True)

    for para in iot_u.parameters():
        para.requires_grad = True
    for para in iot_v.parameters():
        para.requires_grad = True
    for para in iot_c.parameters():
        para.requires_grad = True

    iot_u.zero_grad()
    iot_v.zero_grad()
    iot_c.zero_grad()

    iotloss = iot(iot_u, iot_v, iot_c, iotsamples1, iotsamples2)
    iotloss.backward()
    optimizer_c.step()
    optimizer_iu.step()
    optimizer_iv.step()

    iotloss_list.append(iotloss.data.numpy())

    print('iter:', k, 'w_loss:', iotloss.data.numpy())

    if k % 200 == 0:
        true_c = (torch.norm(iotsamples1 - iotsamples2, dim=1)).reshape(-1, 1)
        gen_c = iot_c(torch.abs(iotsamples1 - iotsamples2)).reshape(-1, 1)
        diffl = (torch.abs(true_c ** 2 - gen_c)).data.numpy()
        diffm = (torch.abs((true_c ** 2 - gen_c).mean())).data.numpy()
        diffl_list.append(diffl)
        diffm_list.append(diffm)
        print(diffl)
        print(diffm)
#####
##### learn u,v,pi with c_nn
############retrain and generate figures
im1 = Image.open('source_cp.tiff')
pix1 = im1.load()

Length1 = im1.size[0]
Height1 = im1.size[1]

samples1 = torch.zeros(Length1 * Height1, 3)

count = 0
dim = 3
scale = 255

for k in range(Length1):
    for l in range(Height1):
        samples1[count] = torch.tensor(pix1[k, l])
        count = count + 1

expsamples1 = samples1 / scale

im2 = Image.open('target_cp.tiff')  # Can be many different formats.
pix2 = im2.load()

Length2 = im2.size[0]
Height2 = im2.size[1]

samples2 = torch.zeros(Length2 * Height2, 3)

count = 0

for k in range(Length2):
    for l in range(Height2):
        samples2[count] = torch.tensor(pix2[k, l])
        count = count + 1

expsamples2 = samples2 / scale

from random import sample

net_u = networku(dim, 36, 1)
net_u.apply(weights_init)

net_v = networkv(dim, 36, 1)
net_v.apply(weights_init)

optimizer_u = optim.Adam(net_u.parameters(), lr=1e-3)
optimizer_v = optim.Adam(net_v.parameters(), lr=1e-3)

iter_uv = 20000
iter_f = 20000
batch_size = 2000
# epsilon = 0.5
loss1_list = []

######## training process for dual variables
for i in range(iter_uv):
    x_sample = get_samples(batch_size, 1)
    x_sample = autograd.Variable(x_sample, requires_grad=True)
    y_sample = get_samples(batch_size, 2)
    y_sample = autograd.Variable(y_sample, requires_grad=True)

    for para in net_u.parameters():
        para.requires_grad = True
    for para in net_v.parameters():
        para.requires_grad = True
    for para in iot_c.parameters():
        para.requires_grad = False

    net_u.zero_grad()
    net_v.zero_grad()

    LOSS1 = -loss_fn(net_u, net_v, iot_c, x_sample, y_sample, epsilon)
    LOSS1.backward()
    optimizer_u.step()
    optimizer_v.step()
    loss1_list.append(LOSS1.data.numpy())

    print(-LOSS1.data.numpy())

net_f = networkf(dim, 36, dim)
net_f.apply(weights_init)
optimizer_f = optim.Adam(net_f.parameters(), lr=1e-3)

loss2_list = []

########## training process for optimal map
for j in range(iter_f):
    xx_sample = get_samples(batch_size, 1)
    xx_sample = autograd.Variable(xx_sample, requires_grad=True)
    yy_sample = get_samples(batch_size, 2)
    yy_sample = autograd.Variable(yy_sample, requires_grad=True)

    for para in net_u.parameters():
        para.requires_grad = False
    for para in net_v.parameters():
        para.requires_grad = False
    for para in iot_c.parameters():
        para.requires_grad = False
    for para in net_f.parameters():
        para.requires_grad = True

    net_u.zero_grad()
    net_v.zero_grad()
    net_f.zero_grad()

    LOSS2 = loss_hn(net_u, net_v, net_f, iot_c, xx_sample, yy_sample, epsilon)
    LOSS2.backward()
    optimizer_f.step()
    loss2_list.append(LOSS2.data.numpy())

    print('iter:', j, 'w_loss:', LOSS2.data.numpy())

    ############### save figures
    if j % 200 == 0:
        plot_samples_traj(j)
        color_transf(j)