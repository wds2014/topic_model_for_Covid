# load data
import numpy as np
import scipy.io as sio

from scipy import sparse as sp
import time
from logger import Logger
#import cPickle as pickle
import pickle
import json
import os
from logger import Logger
logger = Logger('./logs')

# data        = sio.loadmat('./mnist_gray')
# train_data  = np.array(np.ceil(data['train_mnist']*5), order ='C')   #0-1    V*N
# test_data   = np.array(np.ceil(data['test_mnist']*5), order = 'C')   #0-1    V*N
# train_label = data['train_label']
# test_label  = data['test_label']

# import h5py
# data_path = './flicker_data/flicker_labeled_split_10_epoch_5.mat'
# data = h5py.File(data_path)
# # data = sio.loadmat(data_path)

# image_labeled_feature = data['image_labeled'][:]  # add [:] change to array
# text_labeled_feature = data['text_labeled'][:]
# tr_te_index = [data[element[0]]['tr_te_index'][:] for element in data['flicker_splits']]
# tr_index = [data[element[0]]['tr_index'][:] for element in data['flicker_splits']]
# te_index = [data[element[0]]['te_index'][:] for element in data['flicker_splits']]

# split_index = 1
# epoch_index = 1
# Setting = {}
# Setting['split_index'] = split_index
# Setting['epoch_index'] = epoch_index
# Setting['tr_index'] = (tr_index[split_index - 1] - 1)
# Setting['tr_index'] = Setting['tr_index'].reshape(Setting['tr_index'].size) # 0 - 24999
# Setting['te_index'] = (te_index[epoch_index - 1] - 1)  # 0 - 24999
# Setting['te_index'] = Setting['te_index'].reshape(Setting['te_index'].size)

# ## normlization
# image_labeled_min = image_labeled_feature.min(1, keepdims=1)
# image_labeled_max = image_labeled_feature.max(1, keepdims=1)
# image_labeled_norm = (image_labeled_feature - image_labeled_min) / (image_labeled_max - image_labeled_min)

# X_img_all = np.round(image_labeled_norm * 25)
# X_txt_all = np.round(text_labeled_feature) * 150 #+ 1


# ## inital
# X_img_tr = np.array(X_img_all[:,Setting['tr_index'].astype('int')],order ='C')
# X_txt_tr = np.array(X_txt_all[:,Setting['tr_index'].astype('int')] ,order ='C')

# X_img_te = np.array(X_img_all[:,Setting['te_index'].astype('int')],order ='C')
# X_txt_te = np.array(X_txt_all[:,Setting['te_index'].astype('int')] ,order ='C')

# V_img = X_img_tr.shape[0]
# N_train = X_txt_tr.shape[1]
# V_txt = X_txt_all.shape[0]


# # X = X_img_tr
# X_sum = np.sum(X_txt_tr, axis=0)
# ind = np.where(X_sum>9*150)
# train_data = X_img_tr[:, ind[0]]

# dim_x, num_x = train_data.shape


train_data = sp.load_npz('./cord19_40000.npz')
train_data = train_data.toarray()
np.random.seed(2018)
np.random.shuffle(train_data)
train_data = train_data.T

dim_x, num_x = train_data.shape


# setting
Setting = {}
Setting['V'] = train_data.shape[0]
Setting['K'] = [128, 64, 32]
Setting['H'] = [128, 64, 32]
Setting['N'] = train_data.shape[1]
Setting['Num_class'] = 10
Setting['Num_layers']= len(Setting['K'])

# online setting
Setting['SweepTimes'] = 20000
Setting['Minibatch'] = 200
Setting['Burnin']  = 5
Setting['Collection'] = 5                                  #
Setting['Iterall'] = Setting['SweepTimes'] * train_data.shape[1]/Setting['Minibatch']
Setting['tao0FR'] = 0
Setting['kappa0FR'] = 0.9
Setting['tao0'] = 20
Setting['kappa0'] = 0.7
Setting['epsi0'] = 1
Setting['FurCollapse'] = 1  # 1 or 0
Setting['flag'] = 0

# load setting
V = Setting['V']
H = Setting['H']
K = Setting['K']
N = Setting['N']
T = len(Setting['K'])
real_min = np.float64(2.2e-308)


# superparams
Supara = {}
Supara['ac'] = 1            ; Supara['bc'] = 1
Supara['a0pj'] = 0.01       ; Supara['b0pj'] = 0.01
Supara['e0cj'] = 1          ; Supara['f0cj'] = 1
Supara['e0c0'] = 1          ; Supara['f0c0'] = 1
Supara['a0gamma'] = 1       ; Supara['b0gamma'] = 1
Supara['eta'] = np.ones(T)*0.1  # 0.01

# params
Phi = [] ; Eta = []
for t in range(T): # 0:T-1
    Eta.append(Supara['eta'][t])
    if t == 0:
        Phi.append(0.2 + 0.8 * np.float64(np.random.rand(V, K[t])))
    else:
        Phi.append(0.2 + 0.8 * np.float64(np.random.rand(K[t-1], K[t])))
    Phi[t] = Phi[t] / np.maximum(real_min, Phi[t].sum(0)) # maximum every elements
r_k = np.ones([K[T-1],1])/K[T-1]    ;  gamma0 = 1 ;  c0 = 1

NDot = [0] * T
Xt_to_t1 = [0] * T
WSZS = [0] * T
EWSZS = [0] * T

ForgetRate = np.power((Setting['tao0FR'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])),-Setting['kappa0FR'])
epsit = np.power((Setting['tao0'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])), -Setting['kappa0'])
epsit = Setting['epsi0'] * epsit / epsit[0]

# define layer
import tensorflow as tf

def log_max(input_x):
    return tf.log(tf.maximum(input_x,real_min))

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float64))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float64))

def encoder_left(input_x,i):   # i = 0:T-1 , input_x N*V
    # params
    H_dim = [V] + H
    W_h = weight_variable(shape=[H_dim[i],H_dim[i+1]])
    b_h = bias_variable(shape=[H_dim[i+1]])

    # feedforwThetaard
    if i == 0:
        output = tf.nn.softplus(tf.matmul(log_max(1 + input_x), W_h) + b_h)   # none * H_dim[i+1]
    else:
        output = tf.nn.softplus(tf.matmul(input_x, W_h) + b_h)               # none * H_dim[i+1]
    return output

def encoder_right(input_x, i, phi, theta): # i = 0:T-1 , input_x N*V
    # params
    H_dim = [V] + H
    K_dim = K
    W_k = weight_variable(shape=[H_dim[i+1], 1])          # params k   H_dim*1
    b_k = bias_variable(shape=[1])
    W_l = weight_variable(shape=[H_dim[i+1], K_dim[i]])   # params l   H_dim*K_dim
    b_l = bias_variable(shape=[K_dim[i]])

    # feedforward
    k_tmp = tf.reshape(tf.maximum(tf.exp(tf.matmul(input_x, W_k) + b_k), real_min),[-1, 1])   # none * 1
    k_tmp = tf.tile(k_tmp, [1, K_dim[i]])                                                     # none * K_dim[i]
    l = tf.maximum(tf.exp(tf.matmul(input_x, W_l) + b_l), real_min)                           # none * K_dim[i]

    if i != len(K) - 1:
#         k = tf.maximum(k_tmp + tf.transpose(tf.matmul(phi, theta)),real_min)                  # none * K_dim[i]
        k = tf.maximum(k_tmp ,real_min)                                                       # none * K_dim[i]
    else:
        k = tf.maximum(k_tmp ,real_min)                                                       # none * K_dim[i]

    return tf.transpose(k), tf.transpose(l)   #  K_dim[i] * none

def reparameterization(Wei_shape, Wei_scale, i, batch_size):
    K_dim   = K

    eps     = tf.random_uniform(shape=[np.int32(K_dim[i]), batch_size], dtype=tf.float64)    # none * K_dim[i]
    theta   = Wei_scale * tf.pow(-log_max(1-eps), 1/Wei_shape)
    return theta   # K_dim[i] * none



def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale): # K_dim[i] * none
    eulergamma = 0.5772

    KL_Part1 = eulergamma * (1 - 1/Wei_shape) + log_max(Wei_scale/Wei_shape) + 1 + Gam_shape * log_max(Gam_scale)
    KL_Part2 = -tf.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma/Wei_shape)
    KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(tf.lgamma(1 + 1/Wei_shape))
    return KL

# build graph
Batch_Size = tf.placeholder(tf.int32)

input_x = tf.placeholder(tf.float64, shape=[None,V])   # N*V
x_vn = tf.transpose(input_x)

phi1 = tf.placeholder(tf.float64, shape = [V, K[0]])
phi2 = tf.placeholder(tf.float64, shape = [K[0], K[1]])
phi3 = tf.placeholder(tf.float64, shape = [K[1], K[2]])

# upward
h_1 = encoder_left(input_x, 0)
h_2 = encoder_left(h_1, 1)
h_3 = encoder_left(h_2, 2)

# downward
k3, l3 = encoder_right(h_3, 2, 0 , 0)
theta3 = reparameterization(k3,l3,2,Batch_Size)
k2, l2 = encoder_right(h_2, 1, phi3, theta3)
theta2 = reparameterization(k2,l2,1,Batch_Size)
k1, l1 = encoder_right(h_1, 0, phi2, theta2)
theta1 = reparameterization(k1,l1,0,Batch_Size)

# loss
Theta1Scale_prior = 1.0
Theta2Shape_prior = 0.01
Theta2Scale_prior = 1.0
theta3_KL = tf.reduce_sum(KL_GamWei(np.float64(0.01), np.float64(1.0), k3, l3))
theta2_KL = tf.reduce_sum(KL_GamWei(tf.matmul(phi3,theta3), np.float64(Theta2Shape_prior), k2, l2))
theta1_KL = tf.reduce_sum(KL_GamWei(tf.matmul(phi2,theta2), np.float64(Theta1Scale_prior), k1, l1))

tmp1 = x_vn * log_max(tf.matmul(phi1, theta1))
tmp2 = tf.matmul(phi1, theta1)
tmp3 = tf.lgamma( x_vn + 1)
Likelihood = tf.reduce_sum( x_vn * log_max(tf.matmul(phi1, theta1)) - tf.matmul(phi1, theta1) - tf.lgamma( x_vn + 1))
Loss       = (0.001*theta3_KL + 0.01*theta2_KL + 0.1*theta1_KL + Likelihood) / tf.to_double(Batch_Size) # * N
LB         = theta3_KL + theta2_KL + theta1_KL + Likelihood
train_step = tf.train.AdamOptimizer(0.0001).minimize(-Loss)



import PGBN_sampler

def updatePhi(miniBatch, Phi, Theta, MBratio, MBObserved):

    Xt = miniBatch   # V*N

    for t in range(len(Phi)):   # t = 0:T-1
        if t == 0:
            Xt_to_t1[t], WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), Phi[t], Theta[t])
        else:
            Xt_to_t1[t], WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(Xt_to_t1[t - 1], Phi[t], Theta[t])

        EWSZS[t] = MBratio * WSZS[t]   # Batch_Num * WSZS[t]

        if (MBObserved == 0):
            NDot[t] = EWSZS[t].sum(0)
        else:
            NDot[t] = (1 - ForgetRate[MBObserved]) * NDot[t] + ForgetRate[MBObserved] * EWSZS[t].sum(0)  # 1*K

        tmp = EWSZS[t] + Eta[t]  # V*K
        tmp = (1 / NDot[t]) * (tmp - tmp.sum(0) * Phi[t])  # V*K
        tmp1= (2 / NDot[t]) * Phi[t]
        tmp = Phi[t] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi[t].shape[0], Phi[t].shape[1])
        Phi[t] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[t], 0)

    return Phi


# Initial
tf.set_random_seed(seed=0)
np.random.seed(seed=0)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train phase
LB_list = []
import time
for sweepi in range(Setting['SweepTimes']):

    idxall = np.linspace(0,N-1,N)
    np.random.shuffle(idxall)

    N_batch = Setting['Minibatch']
    MBratio = np.floor(N/N_batch).astype('int')
    Loss_t = 0
    Likelihood_t = 0
    time1 = time.time()
    for MBt in range(MBratio):

        MBObserved = (sweepi*MBratio + MBt).astype('int')
        if (MBObserved == Setting['Iterall']):
            Setting['flag'] = 1
            break
        MB_index = idxall[MBt*N_batch + np.arange(N_batch)].astype('int')
        X_batch  = np.array(train_data[:,MB_index], order='C').astype('double')

        train_step.run(feed_dict = {input_x:np.transpose(X_batch), phi1:Phi[0], phi2:Phi[1], phi3:Phi[2], Batch_Size:N_batch})
        Theta = sess.run([theta1, theta2, theta3], feed_dict = {input_x:np.transpose(X_batch),
                                                                phi1:Phi[0], phi2:Phi[1], phi3:Phi[2],
                                                                Batch_Size:N_batch})

        Phi = updatePhi(X_batch, Phi, Theta, MBratio, MBObserved)
        Loss_t += sess.run(Loss, feed_dict={input_x: np.transpose(X_batch), phi1: Phi[0], phi2: Phi[1], phi3: Phi[2],
                                           Batch_Size: N_batch})
        Likelihood_t += sess.run(Likelihood,
                                feed_dict={input_x: np.transpose(X_batch), phi1: Phi[0], phi2: Phi[1], phi3: Phi[2],
                                           Batch_Size: N_batch})
    time2 = time.time()
    logstr = "Epoch:{:3d}   Training Likelihood:{:<12.2f}   Training Loss: {:<12.2f}  Cost: {:<2.1f} s".format(
        sweepi, Likelihood_t, Loss_t, time2 - time1)
    print(logstr)
    info = {
    'loss':Loss_t,
     'likelihood':Likelihood_t
    }
    for tag,value in info.items():
        logger.scalar_summary(tag,value,sweepi)
    if np.mod(sweepi, 100) == 0:
        Theta = sess.run([theta1, theta2, theta3], feed_dict = {input_x:np.transpose(train_data),
                                                                phi1:Phi[0], phi2:Phi[1], phi3:Phi[2],
                                                                Batch_Size:Setting['N']})
        
        np.save('./Theta.npy',Theta)
        np.save('./Phi.npy',Phi)
np.save('./Theta.npy',Theta)
np.save('./Phi.npy',Phi)
        #sio.savemat("Theta.mat", {"Theta": Theta})
        # N_test = test_data.shape[1]
        # Loss_t = sess.run(Loss,feed_dict = {input_x:np.transpose(test_data), phi1:Phi[0], phi2:Phi[1], phi3:Phi[2],Batch_Size:N_test})
        # LB_t = sess.run(LB,feed_dict = {input_x:np.transpose(test_data), phi1:Phi[0], phi2:Phi[1], phi3:Phi[2],Batch_Size:N_test})
        # Likelihood_t = sess.run(Likelihood,feed_dict = {input_x:np.transpose(test_data), phi1:Phi[0], phi2:Phi[1], phi3:Phi[2],Batch_Size:N_test})
        # print sweepi, Loss_t, LB_t, Likelihood_t

