# -*- coding: utf-8 -*-
####################################################
#         ╔══╗                                     #
#         ╚╗╔╝                                     #
#         ╔╝(¯`v´¯)                                #
#         ╚══`.¸.Coding ~                          #
#                                                  #
# @Author: wang.dongsheng                          #
# @E-mail: hellowds2014@gmail.com                  #
# @Date:   2019-05-24 11:40:56                     #
# @Last Modified by:   wang.dongsheng              #
# @Last Modified time: 2019-05-24 11:45:50        #
####################################################
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import sparse as sp
import time
from logger import Logger
logger = Logger('../logs')
import cPickle as pickle
import json
import os

## function
import PGBN_sampler
realmin = 2.2e-308

train_data = sp.load_npz('../data_sparse.npz')
train_data = train_data.toarray()
np.random.seed(2018)
np.random.shuffle(train_data)
train_data = train_data.T
voc = np.load('../vocab.npy')
train_data = np.array(train_data,order = 'C')
print(train_data.shape)

K = np.array([1328,263,16])
T = K.size              # length of matrix

Supara = {}             # an empty dict
Supara['ac'] = 1            ; Supara['bc'] = 1
Supara['a0pj'] = 0.01       ; Supara['b0pj'] = 0.01
Supara['e0cj'] = 1          ; Supara['f0cj'] = 1
Supara['e0c0'] = 1          ; Supara['f0c0'] = 1
Supara['a0gamma'] = 1       ; Supara['b0gamma'] = 1
Supara['eta'] = np.ones(T)*0.01

Setting = {}
Setting['Iterall'] = 400

## inital

V_train = train_data.shape[0]
N_train = train_data.shape[1]

Phi = []    ; Eta = []
# Phi_new=[]
# Phi_new.append(phi0)
# Phi_new.append(phi1)
# Phi_new.append(phi2)
# for t in range(T):
#     Eta.append(Supara['eta'][t])
for t in range(T):  # 0-T-1
    Eta.append(Supara['eta'][t])
    if t==0:
        Phi.append(0.2+0.8*np.random.rand(V_train,K[t]))
#         Phi[0][392,0]=10.0
#         Phi[0][219,0]=10.0
#         Phi[0][501,0]=10.0
#         Phi[0][459,0]=10.0
#         Phi[0][167,1]=10.0
#         Phi[0][126,1]=10.0
#         Phi[0][128,1]=10.0
#         Phi[0][170,1]=10.0
#         Phi[0][15,2]=10.0
#         Phi[0][119,2]=10.0
#         Phi[0][244,2]=10.0
#         Phi[0][34,2]=10.0
#         Phi[0][88,3]=10.0
#         Phi[0][697,3]=10.0
#         Phi[0][406,3]=10.0


    else:
        Phi.append(0.2+0.8*np.random.rand(K[t-1],K[t]))
    Phi[t] = Phi[t] / np.maximum(realmin,Phi[t].sum(0))  # maximum every elements
r_k = np.ones([K[T-1],1])/K[T-1]    ;   gamma0 = 1 ; c0 = 1

Theta = []      ;   c_j = []
for t in range(T):
    Theta.append(np.ones([K[t],N_train])/K[t])
    c_j.append(np.ones([1,N_train]))
c_j.append(np.ones([1,N_train]))    ;   p_j = PGBN_sampler.Calculate_pj(c_j,T) # need to mix up
Xt_to_t1 = []   ;   WSZS = []
for t in range(T):
    Xt_to_t1.append(np.zeros(Theta[t].shape))
    WSZS.append(np.zeros(Phi[t].shape))

for my_iter in range(Setting['Iterall']):

    start_time = time.time()
    ## Upward Pass
    #  update Phi
    for t in range(T):
        if t == 0:
            Xt = train_data
            Xt_to_t1[t],WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'),Phi[t],Theta[t])
        else:
            Xt_to_t1[t],WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(Xt_to_t1[t-1].astype('double'),Phi[t],Theta[t])

#            Xt1 = PGBN_sampler.Crt_Matrix(Xt_to_t1[t-1].astype('double'),np.dot(Phi[t],Theta[t]))
#            Xt_to_t1[t],WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt1.astype('double'),Phi[t],Theta[t])

        # if t ==0 :
        Phi[t][:,:] = PGBN_sampler.Sample_Phi(WSZS[t],Eta[t])

    ## Downward Pass
    # update c_j

    if my_iter>10:
        if T>1:
            for n in range(N_train):
                p_j[1][0,n] = np.random.beta(Xt_to_t1[0][:,n].sum(0)+Supara['a0pj'], Theta[1][:,n].sum(0)+Supara['b0pj'])
        else:
            for n in range(N_train):
                p_j[1][0,n] = np.random.beta(Xt_to_t1[0][:,n].sum(0)+Supara['a0pj'], r_k.sum(0)+Supara['b0pj'])
        p_j[1] = np.minimum(np.maximum(p_j[1],realmin),1-realmin)  ## make sure p_j is no so large or so small
        c_j[1][:,:] = (1-p_j[1])/p_j[1]

        for t in [i for i in range(T+1) if i>1]:    # only T>=2 works  ==> for t = 3:T+1
            if t == T:
                for n in range(N_train):
                    c_j[t][0,n] = np.random.gamma(r_k.sum(0) + Supara['e0cj'],1)/(Theta[t-1][:,n].sum(0) + Supara['f0cj'])
            else:
                for n in range(N_train):
                    c_j[t][0,n] = np.random.gamma(Theta[t][:,n].sum(0) + Supara['e0cj'],1)/(Theta[t-1][:,n].sum(0) + Supara['f0cj'])
        p_j_tmp = PGBN_sampler.Calculate_pj(c_j,T)
        p_j[2:] = p_j_tmp[2:]

    # update theta
    for t in range(T-1,-1,-1):   ## for t = T:-1 :1
        if t == T-1:
            shape = np.repeat(r_k,N_train,axis=1)
        else:
            shape = np.dot(Phi[t+1],Theta[t+1])
        Theta[t] = PGBN_sampler.Sample_Theta(Xt_to_t1[t],c_j[t+1],p_j[t],shape)

    end_time = time.time()

    print "epoch " + str(my_iter) + " takes " + str(end_time - start_time) + " seconds"
    print "error " + str(PGBN_sampler.Reconstruct_error(Xt,Phi[0],Theta[0]))
    LH = np.sum(train_data*np.log(np.dot(Phi[0],Theta[0])) - np.dot(Phi[0],Theta[0]))
    print "Likelihood " + str(LH/1000)
    error_rec=PGBN_sampler.Reconstruct_error(Xt,Phi[0],Theta[0])
    info = {
    'error':error_rec,
     'likelihood':LH/1000
    }
    for tag,value in info.items():
        logger.scalar_summary(tag,value,my_iter)
