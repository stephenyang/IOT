
import numpy as np
import matplotlib.pyplot as plt
import ot

import matplotlib
matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)


## IOT
def Sinkhorn_IOT(p1,p2,pi,epsilon,maxiter):
    alpha = np.random.rand(n,1)
    beta = alpha
    # epsilon = gamma#0.1
    cost = np.random.rand(n,n)
    cost = 0.5*(cost + np.transpose(cost))
    cost = cost - np.diag(np.diag(cost))
    u = np.exp(alpha/epsilon)
    v = np.exp(beta/epsilon)
    p1= p1.reshape((n,1))
    p2= p2.reshape((n,1))

    c_err = np.zeros((maxiter,1))
    obj = np.zeros((maxiter,1))

    for iter in range(maxiter):
        K = np.exp(-cost/epsilon)#
        u = p1/np.matmul(K,v)
        # v = p2/np.matmul(np.transpose(K),u)
        v = p2 / np.matmul(K.T, u)
        K = np.power(u,-1)*pi*np.power(np.transpose(v),-1)
        # print(iter)
        cost = -epsilon * np.log(K)#
        cost = 0.5 * (cost + np.transpose(cost))
        cost = cost - np.diag(np.diag(cost))
        cost = np.maximum(cost, 0)

        c_err[iter] = np.linalg.norm(c-cost,'fro')/np.linalg.norm(cost,'fro')
        a = epsilon * np.log(u)
        b = epsilon * np.log(v)
        s = 0
        for i in range(len(a)):
            for j in range(len(b)):
                s = s + np.exp((a[i]+b[j]-cost[i,j])/epsilon)

        obj[iter] = -np.dot(p1.T,a)-np.dot(p2.T,b)+np.dot(cost,pi).sum() + epsilon * s

    return a,b,cost,c_err,obj


## numerical expertiments
colors=['k','b','g','r']

n = 100
x = np.linspace(1/n, 1, num=n)

p1 = 0.55 * ot.datasets.get_1D_gauss(n,20,18) + 0.45 * ot.datasets.get_1D_gauss(n,80,9)
p2 = 0.55 * ot.datasets.get_1D_gauss(n,15,9) + 0.45 * ot.datasets.get_1D_gauss(n,65,15)
# plt.plot(x,p1,'o-',color='blue')
# plt.plot(x,p2,'o-',color='red')
# plt.tight_layout()
# plt.show()
# temp = -1

### varying p
for p in [0.5,1,2,3]:
    temp+=1
    print(temp)
    c = np.absolute(x-np.expand_dims(x,axis=1))
    # c = c/n
    c = np.power(c,p)
    # print(c)

    gamma = 0.1
    K = np.exp(-c/gamma)
    u = np.ones([n,])
    v = u
    maxiter = 500
    for iter in range(maxiter):
        u = p1/np.matmul(K,v)
        v = p2/np.matmul(np.transpose(K),u)
    pi = np.expand_dims(u,axis=1)*K*np.expand_dims(v,axis = 0) # + np.finfo(float).eps
    # plt.imshow(pi) # plot transport plan
    # plt.show()
    #
    # plt.plot(np.sum(pi,axis = 0))
    # plt.plot(np.sum(pi,axis = 1))
    # plt.show()
    [a, b, cost, c_err, obj] = Sinkhorn_IOT(p1,p2,pi,gamma,maxiter)
    plt.figure(1)
    plt.semilogy(c_err,colors[temp])
    plt.title('Relative_error', fontsize=25)
    # plt.show()

    plt.figure(2)
    plt.plot(obj,colors[temp])
    plt.title('Obj_val', fontsize=25)

plt.show()


# plt.imshow(c)
# plt.colorbar()
# # plt.clim(0,1)
# plt.title('true c', fontsize=25)
#
# plt.imshow(cost)
# plt.colorbar()
# # plt.clim(0,1)
# plt.title('recovered c', fontsize=25)
# plt.show()


# plt.plot(c_err)

### varying epsilon
temp = -1
p = 2
for gamma in [10,1,0.1,0.01]:
    temp+=1
    print(temp)
    c = np.absolute(x-np.expand_dims(x,axis=1))
    # c = c/n
    c = np.power(c,p)
    # print(c)

    # gamma = 0.1
    K = np.exp(-c/gamma)
    u = np.ones([n,])
    v = u
    maxiter = 500
    for iter in range(maxiter):
        u = p1/np.matmul(K,v)
        v = p2/np.matmul(np.transpose(K),u)
    pi = np.expand_dims(u,axis=1)*K*np.expand_dims(v,axis = 0) # + np.finfo(float).eps
    # plt.imshow(pi) # plot transport plan
    # plt.show()
    #
    # plt.plot(np.sum(pi,axis = 0))
    # plt.plot(np.sum(pi,axis = 1))
    # plt.show()
    [a, b, cost, c_err, obj] = Sinkhorn_IOT(p1,p2,pi,gamma,maxiter)
    plt.figure(3)
    plt.semilogy(c_err,colors[temp])
    plt.title('Relative_error', fontsize=25)
    # plt.show()

    # plt.figure(2)
    # plt.plot(obj,colors[temp])
    # plt.title('Obj_val', fontsize=25)

plt.show()