# -*- coding: utf-8 -*-
####################################################
#         ╔══╗                                     #
#         ╚╗╔╝                                     #
#         ╔╝(¯`v´¯)                                #
#         ╚══`.¸.Coding ~                          #
#                                                  #
# @Author: wang.dongsheng                          #
# @E-mail: hellowds2014@gmail.com                  #
# @Date:   2019-05-24 16:31:50                     #
# @Last Modified by:   wang.dongsheng              #
# @Last Modified time: 2019-05-24 18:38:49        #
####################################################
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import sparse as sp
import time
from logger import Logger
#import cPickle as pickle
import pickle
import json
import os
from dic_top_vision import dic_topic_vision
from topic_tree import plot_tree
## function
import PGBN_sampler
realmin = 2.2e-308

Supara = {}             # an empty dict
Supara['ac'] = 1            ; Supara['bc'] = 1
Supara['a0pj'] = 0.01       ; Supara['b0pj'] = 0.01
Supara['e0cj'] = 1          ; Supara['f0cj'] = 1
Supara['e0c0'] = 1          ; Supara['f0c0'] = 1
Supara['a0gamma'] = 1       ; Supara['b0gamma'] = 1


class PGBN():
    """
    Thanks for chaojie
    """
    def __init__(self, train_data, K = [128,64,32], voc=None, save_fre=100):
        """
        train_data : V * N
        K : default [128,64,32]
        iter : default 400
        """
        self.train_data = np.array(train_data, order = 'C')
        self.V_train, self.N_train = self.train_data.shape
        self.K = np.array(K)
        self.T = self.K.size
        self.Supara = Supara
        self.Supara['eta'] = np.ones(self.T)*0.01
        self.init_param()
        self.save_fre = save_fre
        if voc is not None: 
            self.voc = voc

    def init_param(self):
        """
        init paraments for train
        """
        self.Phi = []
        self.Eta = []
        for t in range(self.T):  # 0-T-1
            self.Eta.append(self.Supara['eta'][t])
            if t==0:
                self.Phi.append(0.2+0.8*np.random.rand(self.V_train,self.K[t]))
            else:
                self.Phi.append(0.2+0.8*np.random.rand(self.K[t-1],self.K[t]))
            self.Phi[t] = self.Phi[t] / np.maximum(realmin,self.Phi[t].sum(0))  # maximum every elements
        self.r_k = np.ones([self.K[self.T-1],1])/self.K[self.T-1]
        self.gamma0 = 1
        self.c0 = 1

        self.Theta = []
        self.c_j = []
        for t in range(self.T):
            self.Theta.append(np.ones([self.K[t],self.N_train])/self.K[t])
            self.c_j.append(np.ones([1,self.N_train]))
        self.c_j.append(np.ones([1,self.N_train]))
        self.p_j = PGBN_sampler.Calculate_pj(self.c_j, self.T) # need to mix up
        self.Xt_to_t1 = []      #sum v-dim among the argument matrix : k * N
        self.WSZS = []          #sum N-dim among the argument matrix : V * K
        for t in range(self.T):
            self.Xt_to_t1.append(np.zeros(self.Theta[t].shape))
            self.WSZS.append(np.zeros(self.Phi[t].shape))

    def train(self, outpath, iteration = 400):
        """
            train PGBN
            save Phi, Theta
        """
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        PGBN_log = outpath + '/logs/'
        if not os.path.exists(PGBN_log):
            os.mkdir(PGBN_log)
        logger = Logger(PGBN_log)
        for my_iter in range(iteration):
            start_time = time.time()
            ## Upward Pass
            #  update Phi
            for t in range(self.T):
                if t == 0:
                    Xt = self.train_data
                    self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), self.Phi[t], self.Theta[t])
                else:
                    self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(self.Xt_to_t1[t-1].astype('double'), self.Phi[t], self.Theta[t])
                self.Phi[t][:,:] = PGBN_sampler.Sample_Phi(self.WSZS[t], self.Eta[t])

            ## Downward Pass
            # update c_j

            if my_iter > 10:
                if self.T>1:
                    for n in range(self.N_train):
                        self.p_j[1][0,n] = np.random.beta(self.Xt_to_t1[0][:,n].sum(0)+self.Supara['a0pj'], self.Theta[1][:,n].sum(0)+self.Supara['b0pj'])
                else:
                    for n in range(self.N_train):
                        self.p_j[1][0,n] = np.random.beta(self.Xt_to_t1[0][:,n].sum(0)+self.Supara['a0pj'], self.r_k.sum(0)+self.Supara['b0pj'])
                self.p_j[1] = np.minimum(np.maximum(self.p_j[1],realmin),1-realmin)  ## make sure p_j is no so large or so small
                self.c_j[1][:,:] = (1-self.p_j[1])/self.p_j[1]

                for t in [i for i in range(self.T+1) if i>1]:    # only T>=2 works  ==> for t = 3:T+1
                    if t == self.T:
                        for n in range(self.N_train):
                            self.c_j[t][0,n] = np.random.gamma(self.r_k.sum(0) + self.Supara['e0cj'],1)/(self.Theta[t-1][:,n].sum(0) + self.Supara['f0cj'])
                    else:
                        for n in range(self.N_train):
                            self.c_j[t][0,n] = np.random.gamma(self.Theta[t][:,n].sum(0) + self.Supara['e0cj'],1)/(self.Theta[t-1][:,n].sum(0) + self.Supara['f0cj'])
                p_j_tmp = PGBN_sampler.Calculate_pj(self.c_j,self.T)
                self.p_j[2:] = p_j_tmp[2:]

            # update theta
            for t in range(self.T-1,-1,-1):   ## for t = T:-1 :1
                if t == self.T-1:
                    shape = np.repeat(self.r_k, self.N_train,axis=1)
                else:
                    shape = np.dot(self.Phi[t+1], self.Theta[t+1])
                self.Theta[t] = PGBN_sampler.Sample_Theta(self.Xt_to_t1[t], self.c_j[t+1], self.p_j[t], shape)

            end_time = time.time()
            error = PGBN_sampler.Reconstruct_error(self.train_data, self.Phi[0], self.Theta[0])
            likelihood = np.sum(self.train_data*np.log(realmin + np.dot(self.Phi[0], self.Theta[0])) - np.dot(self.Phi[0], self.Theta[0]))
            info = {
                    'error':error,
                    'likelihood':likelihood/1000
                    }
            for tag,value in info.items():
                logger.scalar_summary(tag,value,my_iter)

            if my_iter % 50 == 0:
                print('{} | {} time : {} error : {} likelihood : {}'.format(my_iter,
                        iteration, end_time-start_time, error, likelihood))
            if my_iter % self.save_fre and self.voc is not None:
                with open(outpath + '/Phi.pick','wb') as f:
                    pickle.dump(self.Phi, f)
                with open(outpath + '/Theta.pick','wb') as f:
                    pickle.dump(self.Theta, f)
                self.Phi_vis()
                # self.show_tree()
        with open(outpath + '/Phi.pick','wb') as f:
            pickle.dump(self.Phi, f)
        with open(outpath + '/Theta.pick','wb') as f:
            pickle.dump(self.Theta, f)
        print('sucessful save Phi, Theta in {}'.format(outpath))
	
    def Phi_vis(self, outpath='output', top_n = 20):
        """
        visualize phi into .txt
        """
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        phi_vis = dic_topic_vision(self.Theta, self.Phi, self.voc)
        phi_vis.vision_dic(outpath,top_n)
    
    def show_tree(self, topic_id=0,threshold=0.01, num=20):
        ## show topic tree for topoic model
        ## return graph, call the graph.render('tree') function will 
        ## aotu-save tree.pdf file
        return plot_tree(self.Phi, self.voc, topic_id=topic_id, threshold=threshold, num=num)


if __name__ == '__main__':
    train_data = sp.load_npz('./cord19_10000_tfidf.npz')
    train_data = train_data.toarray()
    np.random.seed(2018)
    np.random.shuffle(train_data)
    train_data = train_data.T
    voc = np.load('./voc_10000_tfidf.npy')
    pgbn = PGBN(train_data,K=[10,5,2,2,2,2],voc=voc)
    pgbn.train('./output',iteration =2)
    pgbn.Phi_vis()
    graph = pgbn.show_tree()
    graph.render('output/tree')
