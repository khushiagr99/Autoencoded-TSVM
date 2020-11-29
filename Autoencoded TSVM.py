#!/usr/bin/env python
# coding: utf-8

import scipy.io
import pandas as pd
import numpy as np
import sklearn
import csv
import tensorflow as tf
import cv2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Flatten,
    Dense
)
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import to_categorical

import os
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow.keras.models as md

# Loading the pre-trained VGG19 model for feature extraction
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

"""Denoising Autoencoder (DAE)"""

get_ipython().run_line_magic('matplotlib', 'inline')

# settings for reproducibility
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

os.environ['TF_DETERMINISTIC_OPS'] = '1'

(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()

train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

def lrelu(x, alpha=0.1):
    return tf.math.maximum(alpha*x, x)

# The encoder pipeline has 2 convolutional and 2 max pooling layers. 
# Pooling layers are used for down sampling of the images.
encoder = Sequential([
    # convolution
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv1'
    ),
    # the input size is 28x28x32
    MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        name='pool1'
    ),
    # the input size is 14x14x32
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv2'
    ),
    # the input size is 14x14x32
    MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        name='encoding'
    )
    # the output size is 7x7x32
])

# The decoder pipeline has two convolutional layers and two convolutional transpose layers. 
# The transpose layers are used for upsampling of images.
decoder = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        name='conv3',
        padding='SAME',
        use_bias=True,
        activation=lrelu
    ),
    # upsampling, the input size is 7x7x32
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample1'
    ),
    # upsampling, the input size is 14x14x32
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample2'
    ),
    # the input size is 28x28x32
    Conv2D(
        filters=1,
        kernel_size=(3,3),
        strides=(1,1),
        name='logits',
        padding='SAME',
        use_bias=True
    )    
])


# Uses sigmoid activation function after the decoded images are obtained from encoder-decoder pipeline.
class EncoderDecoderModel(Model):
    def __init__(self, is_sigmoid=False):
        super(EncoderDecoderModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._is_sigmoid = is_sigmoid
       
    def call(self, x):
        x = self._encoder(x)
        decoded = self._decoder(x)
        if self._is_sigmoid:
            decoded = tf.keras.activations.sigmoid(decoded)
        return decoded


# noise_factor is a hyperparameter which can be tuned as per the requirement.
def distort_image(input_imgs, noise_factor=0.5):
    noisy_imgs = input_imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_imgs.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs


train_imgs_data = train_imgs[..., tf.newaxis]
test_imgs_data = test_imgs[..., tf.newaxis]
train_noisy_imgs = distort_image(train_imgs_data)
test_noisy_imgs = distort_image(test_imgs_data)

def cost_function(labels=None, logits=None, name=None):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)
    return tf.reduce_mean(loss)

def plot_losses(results):
    plt.plot(results.history['loss'], 'bo', label='Training loss')
    plt.plot(results.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss',fontsize=14)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.legend()
    plt.show()
    plt.close()


encoder_decoder_model = EncoderDecoderModel()
num_epochs = 25
batch_size_to_set = 64
learning_rate = 1e-5
num_workers = 2


encoder_decoder_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    loss=cost_function,
    metrics=None
)

results = encoder_decoder_model.fit(
    train_noisy_imgs,
    train_imgs_data,
    epochs=num_epochs,
    batch_size=batch_size_to_set,
    validation_data=(test_noisy_imgs, test_imgs_data),
    workers=num_workers,
    shuffle=True
)

plot_losses(results)

encoder_decoder_model2 = EncoderDecoderModel(is_sigmoid=True)
img_num_to_decode = 10
test_imgs_data_decode = test_imgs_data[:]
test_noisy_imgs_decode = tf.cast(test_noisy_imgs[:], tf.float32)
decoded_images = encoder_decoder_model2(test_noisy_imgs_decode)


"""Conventional TSVM (for reconstructed images)"""

# extracts features using pre-trained VGG19 model, followed by PCA.
def feature_extraction(X):
    x_flat=[]
    for i in range(0,X.shape[0]):
        x = X[i]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Predicting features using VGG19
        fea = base_model.predict(x)
        x_flat.append(fea)
    x_flat = np.array(x_flat)
    x_flat = x_flat.reshape(x_flat.shape[0],x_flat.shape[4])
    
    # Employing Principal Component Analysis (PCA) for further refinement in feature extraction.
    pca = PCA().fit(x_flat)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    pca = PCA(n_components=128)
    pca.fit(x_flat)
    X_pca = pca.transform(x_flat)
    return X_pca


# Conventional TSVM using Hinge Loss (non-robust)
def transductive_linear_svm_sg_hinge(XX,yy,L,U,w,b,C1,C2,beta,alphat):
    T = 1500
    n = len(yy)
    unlabeled = XX[:,L:].transpose()
    mean_unlabeled = np.sum(unlabeled,axis = 0)/np.shape(unlabeled)[0]
    mu = np.concatenate((mean_unlabeled,np.asarray(1)), axis = None)
    gamma=np.sum(yy[0:L])/L
    norm_mu = np.dot((mu).transpose(),mu)
    d = np.shape(XX)[0]
    for t in range(T):
        ri = np.random.permutation(range(n))
        alpha = alphat/(t+1)
        for i in range(n):
            ii = ri[i]
            xi = XX[:,ii]
            yi = yy[ii]
            # Calculating score
            score=(np.matmul(w.transpose(),xi)+b)*yi
            
            # Gradients using Hinge Loss Function
            if ii <= L and score < 1:
                gw = (-C1*yi*xi/L)
                gb = (-C1*yi/L)
            elif ii > L and score < 1:
                gw = ((-C2*yi*xi)+(beta[ii-L]*yi*xi))/(2*U)
                gb = ((-C2*yi)+(beta[ii-L]*yi))/(2*U)
            elif ii > L and score >= 1:
                gw = (beta[ii-L]*yi*xi)/(2*U)
                gb = (beta[ii-L]*yi)/(2*U)
            else:
                gw = np.zeros(XX.shape[0])
                gb = 0
            gw = gw.reshape(XX.shape[0],1)
            
            # Updating weights
            w=w-alpha*(w/n+gw)
            b=b-alpha*gb
            
            # Projection onto the constrained space
            ww = np.concatenate((w,b),axis=None)          
            val = np.matmul(mu.transpose(),ww)
            Pww = (mu*(gamma-val)/norm_mu)+ww
            w=Pww[0:d]
            b=Pww[d]
            w = w.reshape(XX.shape[0],1)
        ww = np.concatenate((w,b),axis=None)          
        val = np.matmul(mu.transpose(),ww)
        Pww = (mu*(gamma-val)/norm_mu)+ww
        w=Pww[0:d]
        b=Pww[d]
        w = w.reshape(XX.shape[0],1)
        
        # Updated scores
        mm = np.matmul(XX.transpose(),w)
        ma = mm + b
        scores = np.multiply(ma.transpose(),yy.transpose())
        scores = scores.transpose()
        ll = np.where(scores<1)
        ll = ll[0]
        kk = np.where(ll <= L)
        kk = kk[0]
        sum_scores = 0
        for a in kk:
            sum_scores += 1 - scores[ll[a]]
        c1 = sum_scores*C1/L
        kk = np.where(ll > L)
        kk = kk[0]
        sum_scores1 = 0
        for a in kk:
            sum_scores1 += 1 - scores[ll[a]]
        c2 = sum_scores1*C2/(2*U)
        c3 = np.matmul(np.transpose(beta),scores[L:])/(2*U)
        norm_w = np.matmul(np.transpose(w),w)/2
        
        # Updated cost
        cost = (0.5*norm_w + (c1+c2+c3))
    w = w.reshape(XX.shape[0],1)
    return [w,b,cost]

def train_linear_transductive_svm_sg(X,y,C1,C2,w0,b0,alpha):
    s = -0.2
    unlabeled = np.where(y==-2)
    pos = np.where(y==1)
    neg = np.where(y==-1)
    npos = len(pos[0])
    nneg = len(neg[0])
    L = npos + nneg
    U = len(unlabeled[0])
    
    # Assigning label = 1 and label = -1 for unlabelled samples
    XX = np.concatenate((X[:,pos], X[:,neg]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    XX = np.concatenate((XX, X[:,unlabeled]), axis = 2)
    XX = np.squeeze(XX)
    yy = np.concatenate((np.ones(npos), -np.ones(nneg)), axis = 0)
    yy = np.concatenate((yy, np.ones(U)), axis = 0)
    yy = np.concatenate((yy, -np.ones(U)), axis = 0)
    nn = np.shape(XX)[1]
    beta = np.zeros(2*U)
    mm = np.matmul(XX.transpose(),w0)
    ma = mm + b0
    scores = np.multiply(ma.transpose(),yy.transpose())
    scores = scores.transpose()
    ll = np.where(scores[L:]<s)
    ll = ll[0]
    for a in ll:
        beta[a] = C2
    w = w0
    b = b0
    for i in range(5):
        wp = w
        bp = b
        func = transductive_linear_svm_sg_hinge(XX,yy,L,U,w,b,C1,C2,beta,alpha)
        w = func[0]
        b = func[1]
        cost = func[2]
        beta = np.zeros(2*U)
        mm = np.matmul(XX.transpose(),w)
        ma = mm + b
        scores = np.multiply(ma.transpose(),yy.transpose())
        scores = scores.transpose()
        ll = np.where(scores[L:]<s)
        ll = ll[0]
        for a in ll:
            beta[a] = C2
    return [w,b]

# Multi-class classification using one-against-rest approach
def tsvm_train_one_against_rest(X,y,nbclass,C1,C2,alpha):
    tsvm_models = {}
    for i in range(nbclass):
        ll = np.where(y==-2)
        ld = np.where(y!=-2)
        
        # Creating a new y array by taking class i positive
        yone = (y==i)+(y!=i)*-1
        ll = ll[0]
        ld = ld[0]
        
        # Assigning -2 to unlabelled samples
        for j in ll:
            yone[j] = -2
            
        clf = SVC(kernel='linear')
        clf.fit(np.transpose(np.squeeze(X[:,ld])), yone[ld])
        w0 = np.matmul(np.transpose(clf.support_vectors_),clf.dual_coef_.transpose())
        b0 = clf.intercept_
        hyperplane = train_linear_transductive_svm_sg(X,yone,C1,C2,w0,b0,alpha)
        tsvm_models[i] = hyperplane
    return tsvm_models

# Predicting labels of each sample based on the score obtained using one-against-rest approach, treating each 
# class as positive once. The class corresponding to maximum score is assigned to the sample. 
def tsvm_predict_one_against_rest(test_labels, test_features, svm_models):
    nbclass = len(svm_models)
    n = test_features.shape[1]
    scores_all = []
    for i in range(nbclass):
        svm_model = svm_models[i]
        score1 = np.squeeze(np.matmul(test_features.transpose(),svm_model[0]) + svm_model[1])
        scores_all.append(score1)
    scores_all = np.array(scores_all)
    y_pred = np.argmax(scores_all,axis=0)
    dif = y_pred - test_labels
    misclassifications = np.where(abs(dif)>=0.5)[0].shape[0]
    accuracy = 100 - (misclassifications/len(test_labels))*100
    return accuracy

def get_resized_images(x_train):
    resized = []
    for i in range(x_train.shape[0]):
        img = Image.fromarray(x_train[i],'RGB')
        new_image = img.resize((32, 32))
        resized.append(np.array(new_image))
    return resized


from PIL import Image
nc = 10
samples_pc = 20 # Samples per class, i.e, in each iteration 20 randomly chosen samples of each class are taken.
x_train_noisy = distort_image(x_train[0:(60*samples_pc),:,:])
x_train_noisy = tf.cast(x_train_noisy, tf.float32)
x_train = encoder_decoder_model2(x_train_noisy)
x_train = x_train.numpy()
x_train = np.array(get_resized_images(x_train))
x_train = feature_extraction(x_train)
y_train = y_train[0:(60*samples_pc)]


# Autoencoded TSVM: Hybrid of DAE and TSVM
def autoencoded_tsvm(x_train):
    c1 = 10
    c2 = 2
    alpha = 0.01
    final_accuracy_tsvm = 0
    for it in range(10):
        class_sample = {} # Dict to store index of positive samples for each class
        Xl = [] # X labelled
        yl = [] # y labelled
        Xu = [] # X unlabelled
        yu = [] # y unlabelled
        for i in range(nc):
            class_sample[i] = np.where(y_train==i)
            class_sample[i] = class_sample[i][0]
        class_random = {}
        
        # Selecting random samples for each class.
        for i in range(nc):
            part = class_sample[i]
            class_random[i] = np.array(random.sample(list(part),samples_pc))
            Xl = np.concatenate((Xl, x_train[class_random[i],:]), axis=0) if i>0 else x_train[class_random[i],:]
            yl = np.concatenate((yl, y_train[class_random[i]]), axis=None) if i>0 else y_train[class_random[i]]
            
        # Using the remaining samples as unlabelled.
        for i in range(nc):
            rem = np.array([i for p in class_sample[i] if p not in class_random[i]])
            if(len(rem)>0):
                Xu = np.concatenate((Xu, x_train[rem,:]), axis=0) if i>0 else x_train[rem,:]
                yu = np.concatenate((yu, y_train[rem]), axis=None) if i>0 else y_train[rem]
        
        X = np.concatenate((Xl, Xu), axis=0)
        y = np.concatenate((yl, -2*np.ones(Xu.shape[0])), axis=None)
        X = np.transpose(X)
        Xu = np.transpose(Xu)
        tsvm_models = tsvm_train_one_against_rest(X,y,nc,c1,c2,alpha)
        accuracy_tsvm = rtsvm_predict_one_against_rest(yu,Xu,tsvm_models)
        final_accuracy_tsvm += accuracy_tsvm
    print("TSVM Accuracy:" + (str)(final_accuracy_tsvm/10))

        
autoencoded_tsvm(x_train)



