
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import time
import matplotlib

import ipot
import eot

matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)

X_dim = 1
z_dim = 1

g_dim = 64
batchsize = 200
lr = 1e-3
beta = 0.1
num_proximal = 200

np.random.seed(0)
y = np.random.uniform(0,2,[50000])
y = np.random.normal(3, 0.1, 50000)
z_pool = np.random.uniform(-1,1,[50000])
z_pool = np.random.normal(-1,0.1,[50000])

def next_batch(data,iter,batchsize):
    N = np.size(data,0)
    start = (iter*batchsize%N)
    return data[start:start+batchsize]

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev/3)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 1)
    return tf.Variable(initial)

def plot_hist(data, iter, mb_size):
    bin = 80
    # plt.plot([0., 0., 2., 2.], [0., 0.5, 0.5, 0.], color='darkred', lw=3, label='Ground Truth')
    plt.hist(data, bins=bin, density=True, edgecolor='peru', color='papayawhip', label='Generated Data')
    if iter == 'Final Result':
        plt.legend(fontsize=15, loc=2)
        plt.ylim([0, 0.8])
    plt.title('Plot of  ' + str(iter), fontsize=25)
    plt.show()

# build the model

z = tf.placeholder(tf.float32, shape = [None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, g_dim]),tf.float64)
G_b1 = bias_variable([g_dim])

G_W2 = tf.Variable(xavier_init([g_dim, X_dim]),tf.float64)
G_b2 = bias_variable([X_dim])

def generator(z, G_W1, G_b1, G_W2, G_b2):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    return G_log_prob

X = generator(z, G_W1, G_b1, G_W2, G_b2)

P_vec = tf.placeholder(tf.float32, shape = [batchsize, z_dim])

theta_G = [G_W1, G_b1, G_W2, G_b2]
G_loss = tf.reduce_sum(P_vec*X)
G_solver = (tf.train.RMSPropOptimizer(learning_rate = lr).minimize(G_loss,var_list = theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_list = []
one = np.ones([batchsize,])

print("start training")
start_time = time.time()
for iter in range(1000):
    y_batch = next_batch(y,iter, batchsize)
    z_batch = next_batch(z_pool, iter, batchsize)
    x_batch = sess.run(X,feed_dict = {z:np.expand_dims(z_batch,axis = 1)})
    xtile = x_batch
    ytile = np.expand_dims(y_batch,axis = 0)
    deltaC = xtile - ytile
    C = deltaC*deltaC
    C = C/np.max(C)

    P = ipot.ipot_WD(one, one, C, beta = beta, max_iter = num_proximal, return_loss = False)
    # P = eot.eot_wd(one, one, C, beta=beta, maxiter=num_proximal)
    WD = np.sum(P*C)
    loss_list.append(WD)
    update = 2*np.sum(P*deltaC,axis = 1)

    _,G_loss_cur = sess.run([G_solver,G_loss],
                            feed_dict = {z:np.expand_dims(z_batch,axis = 1),P_vec:np.expand_dims(update,axis = 1)})
    if iter%200==0:
        print('iter:',iter, ' WD:',WD,' Time:', time.time()-start_time)
        data_figure = sess.run(X,feed_dict={z:np.expand_dims(z_pool,axis = 1)})
        plot_hist(data_figure,iter, batchsize)


#plot the loss
plt.plot(loss_list)
plt.title('Plot of WD', fontsize=25)
plt.show()

#plot the generated result
plot_hist(data_figure,'Final Result',batchsize)
