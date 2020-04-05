# -*- coding: utf-8 -*-
####################################################
#         ╔══╗                                     #
#         ╚╗╔╝                                     #
#         ╔╝(¯`v´¯)                                #
#         ╚══`.¸.Coding ~                          #
#                                                  #
# @Author: wang.dongsheng                          #
# @E-mail: hellowds2014@gmail.com                  #
# @Date:   2018-10-24 14:59:26                     #
# @Last Modified by:   wang.dongsheng              #
# @Last Modified time: 2019-03-28 09:05:40        #
####################################################
import os
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

class dic_topic_vision(object):
    '''
        get each layer phi
    '''
    def __init__(self,theta,phi,dic):
        self.theta = theta
        self.phi = phi
        # self.dic = self.get_dic(dic)
        self.dic = dic
        self.dic_len = len(dic)
        self.layers = len(theta)

    def vision_dic(self,outpath,top_n):
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        phi = 1
        for num, phi_layer in enumerate(self.phi):
            phi = np.dot(phi,phi_layer)
            phi_k = phi.shape[1]
            path = os.path.join(outpath, 'phi'+str(num)+'.txt')
            f = open(path,'w')
            for each in range(phi_k):
                top_n_words = self.get_top_n(phi[:,each],top_n)
                f.write(top_n_words)
                f.write('\n')
            f.close()

    def get_top_n(self,phi,top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.dic[index]
            top_n_words += ' '
        return top_n_words

#    def get_dic(self,dic):
#        dic_np = np.load(dic)
#        dic = dic_np.item()
#        my_dic = {}
#        for key in dic.keys():
#            my_dic[dic[key]] = key
#        return my_dic
    def get_dic(self,dic):
        dic_np = dic
        my_dic={}
        for each in range(len(dic_np)):
            my_dic[each] = dic_np[each][0]
        return my_dic

class doc_topic(dic_topic_vision):
    '''
    get each doc  [num_layer,topic_n] topic
    '''
    def __init__(self,theta,phi,dic):
        super(doc_topic,self).__init__(theta,phi,dic)

    def vision_topic(self,outpath,num_layer,theta,topic_n,word_n,k):
        assert num_layer < self.layers, 'you must vision layer in total layers !'
        phi = 1
        for i in range(num_layer):
            phi = np.dot(phi, self.phi[i])
        assert len(theta) == phi.shape[1], 'you must have the right dim !'
        idx = np.argsort(-theta)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        path = os.path.join(outpath, 'class_'+str(k)+'_layer_'+str(num_layer)+'.txt')
        f = open(path,'w')
        for i in range(topic_n):
            f.write(str(idx[i]))
            f.write('\t')
            top_n_words = self.get_top_n(phi[:,idx[i]],word_n)
            f.write(top_n_words)
            f.write('\n')
        f.close()


def get_vision(my_dic_vision, top_n,dic,path):
        top_n_words = ''
        for each in range(my_dic_vision.shape[1]):
            idx = np.argsort(-my_dic_vision[:,each])
            for i in range(top_n):
                idxx = idx[i]
                top_n_words += dic[idxx][0]
                top_n_words += '\t'
            top_n_words += '\n'
        with open(path,'w') as f:
            f.write(top_n_words)

class mnist_vis():
    def __init__(self,theta,phi,fake_label=False,ture_label=False):
        self.theta = np.load(theta)   # K*N
        self.phi = np.load(phi)       # V*K
        self.fake_label = fake_label
        self.ture_label = ture_label
        self.restructure_x = np.dot(self.phi[0],self.theta[0])
        print(self.theta[0].shape)
        print(self.phi[0].shape)
        print(self.restructure_x.shape)

    def get_pic(self,pic_num,save_path):
        pics = self.restructure_x[:,0:pic_num]
        print(pics.shape)
        pics = pics.reshape(28,28,pic_num).astype(np.int16)
        print(pics.shape)

        # plt.imshow(pics[:,:,0],cmap='gray')
        # plt.show()
        cv2.imshow('a',pics[:,:,0])
        cv2.waitKey(0)





if __name__ == '__main__':
    my_path = '/wds_home/cluster_new/mult_layers_1/alfa_20_cluster_20_topic_256_r0_0.1_top_10_information/'

    phi = np.load(my_path+'best_phi.npy')
    theta = np.load(my_path+'best_theta.npy')
    rc=np.load(my_path+'best_rc_.npy')
    dic = np.load('./data/vocab.npy')
    print('done')


    # phi = phi[0]
    theta = theta[0]
    print(phi.shape,theta.shape)

    pih_vis = dic_topic_vision(theta,phi,dic)
    pih_vis.vision_dic('./data/layer_1_topic_256_information',20)

    # phi = np.dot(phi[0],phi[1])  # V*K
    # phi = np.dot(phi_temp,phi[2])
    # theta = np.dot(theta[0],theta[1]) # K*N
    # rc = rc.T # K*C
    # my_dic_vision = np.dot(phi,rc)  #V*C

    # dic = np.load('./voc.npy')
    # path = './my_dic/dic_360_phi.txt'
    # get_vision(my_dic_vision,40,dic,path)




    # phi_path='/wds_home/cluster/mnist_result/alfa_5_cluster_100_topic_256_r0_0.1_top_10/best_phi70.npy'
    # theta_path = '/wds_home/cluster/mnist_result/alfa_5_cluster_100_topic_256_r0_0.1_top_10/best_theta70.npy'
    # rc_path = '/wds_home/cluster/mnist_result/alfa_5_cluster_100_topic_256_r0_0.1_top_10/best_rc_70.npy'
    # phi_path ='/wds_home/cluster/mnist_result/alfa_10_cluster_10_topic_256_r0_0.1_top_10/best_phi35.npy'
    # theta_path='/wds_home/cluster/mnist_result/alfa_10_cluster_10_topic_256_r0_0.1_top_10/best_theta35.npy'
    # rc_path = '/wds_home/cluster/mnist_result/alfa_10_cluster_10_topic_256_r0_0.1_top_10/best_label35.npy'
    # mnist_show = mnist_vis(theta_path,phi_path)
    # mnist_show.get_pic(100,'./mnist/')
    # a=np.array([1,2,5,8,7,4,1,3,2])
    # b=np.argsort(-a)
    # c=np.argsort(a)
    # print(a)
    # print(b)
    # print(c)


#     my_sision = dic_topic_vision('theta.npy','phi.npy','./voc.npy')
#     my_sision.vision_dic('./my_dic',20)

#      my_sision = doc_topic('./theta2050.npy','phi2050.npy','./voc.npy')
#      rc=np.load('./rc_2050.npy')
#      for i in range(len(rc)):
# #         path = './dic_topic'+str(i)
# #         if not os.path.exists(path):
# #             os.mkdir(path)
#          my_sision.vision_topic('./dic_topic',1,rc[i],5,20,i)
