##**Experiments on Adult Dataset with Feature Exemptions**##


import time
import numpy as np
import random
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")
#from IPython import display
#matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
import keras.backend as K
import helpers as H

import os
if not os.path.exists('Results'):
    os.makedirs('Results')


#Define Number of Epochs and Monte-Carlo Iterations
Epochs=300
monte=10


#Creating the Train-Test Dataset
X, y, z, c = H.load_ICU_data('adult.data')

z = np.array(z)
c = np.array(c)
y = np.array(y)

X_train, X_test, y_train, y_test, z_train, z_test, c_train, c_test = train_test_split(X, y, z, c, test_size = 0.3)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Defining the bins for the critical feature hours per week
dm1=20
dm2=30
dm3=40
print(dm1)
print(dm2)
print(dm3)
n_features=X_train.shape[1]

# Defining the ML model
InputZ = Input(shape=(1,))
InputC = Input(shape=(1,))
InputX = Input(shape=(n_features,))
layer1 = Dense(32, activation='relu')(InputX)
dropout1 = Dropout(0.2)(layer1)
layer2 = Dense(32, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(layer2)
layer3 = Dense(32, activation='relu')(dropout2)
dropout3 = Dropout(0.2)(layer3)
predictions = Dense(1, activation='sigmoid')(dropout3)
classifier = Model(inputs=[InputX, InputZ, InputC], outputs=predictions)


# Define three custom loss functions


# Create a loss function for maximum accuracy with no fairness
def bias1(lz,lx):
    def loss1(y_true,y_pred):
        Z = lz
        L1 = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        return L1
    return loss1


# Create a loss function with MI as regularizer
def bias2(lz,lx,gamma):
    def loss2(y_true,y_pred):
        Z = lz
        L1 = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        L2 = H.MI(Z,y_pred)
        return L1 + gamma*L2
    return loss2


# Create a loss function with CMI as regularizer
def bias3(lz,lx,gamma):
    def loss3(y_true,y_pred):
        Z = lz
        C = lx
        L1 = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        L2 = H.CMI(Z,y_pred,C,dm1,dm2,dm3)
        return L1 + gamma*L2
    return loss3



# Repeat multiple times for each loss function for average case results
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


#Loss1
AUC1=[]
Acc1=[]
MI_test1=[]
MI_test_pred1=[]
CMI_test1=[]
CMI_test_pred1=[]

for i in range(monte):
    print('Monte-Carlo Iteration:{}'.format(i))
    reset_weights(classifier)
    classifier.compile(optimizer = 'adam', loss = bias1(InputZ,InputC), metrics = ['accuracy'])
    classifier.fit([X_train, z_train, c_train], y_train, batch_size = 1000, epochs = Epochs,verbose=0)

    # predict on test set
    y_pred = classifier.predict([X_test, z_test, c_test]).ravel()
    AUC1.append(roc_auc_score(y_test, y_pred))
    Acc1.append(100*accuracy_score(y_test, (y_pred>0.5)))
    MI_test1.append(H.brute_force_MI(np.array(z_test), np.array(y_test)))
    MI_test_pred1.append(H.brute_force_MI(np.array(z_test), np.array(y_pred>0.5)))
    CMI_test1.append(H.brute_force_CMI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    CMI_test_pred1.append(H.brute_force_CMI(np.array(z_test), np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))
           

    if i==2:
        fig = H.plot_distributions(y_pred, z_test, fname='Results/no_fairness.pdf')
        fig1 = H.plot_distributions(y_pred[c_test<=dm1], z_test[c_test<=dm1], fname='Results/no_fairness_bin1.pdf')
        fig2 = H.plot_distributions(y_pred[(c_test>dm1)&(c_test<=dm2)], z_test[(c_test>dm1)&(c_test<=dm2)], fname='Results/no_fairness_bin2.pdf')
        fig3 = H.plot_distributions(y_pred[(c_test>dm2)&(c_test<=dm3)], z_test[(c_test>dm2)&(c_test<=dm3)], fname='Results/no_fairness_bin3.pdf')
        fig4 = H.plot_distributions(y_pred[c_test>dm3], z_test[c_test>dm3], fname='Results/no_fairness_bin4.pdf')

        plt.close('all')



print('Accuracy:{np.mean(np.array(AUC1))}')
print('AUC:{np.mean(np.array(Acc1))}')
print('I(Z;Ytrue): {np.mean(np.array(MI_test1))}')
print('I(Z;Ypred):{np.mean(np.array(MI_test_pred1))}')
print('I(Z;Ytrue|Xc): {np.mean(np.array(CMI_test1))}')
print('I(Z;Ypred|Xc): {np.mean(np.array(CMI_test_pred1))}')



#Loss 2 with multiple regularizers
Reg=[2,4]

for j in range(2):
    gamma = Reg[j]
    AUC2=[]
    Acc2=[]
    MI_test2=[]
    MI_test_pred2=[]
    CMI_test2=[]
    CMI_test_pred2=[]
    print('Gamma:{}'.format(gamma))
    for i in range(monte):
        print('Monte-Carlo Iteration:{}'.format(i))
        reset_weights(classifier)
        classifier.compile(optimizer = 'adam', loss = bias2(InputZ,InputC,gamma), metrics = ['accuracy'])
        classifier.fit([X_train, z_train, c_train], y_train, batch_size = 1000, epochs = Epochs,verbose=0)
    
        # predict on test set
        y_pred = classifier.predict([X_test, z_test, c_test]).ravel()
        AUC2.append(roc_auc_score(y_test, y_pred))
        Acc2.append(100*accuracy_score(y_test, (y_pred>0.5)))
        MI_test2.append(H.brute_force_MI(np.array(z_test),np.array(y_test)))
        MI_test_pred2.append(H.brute_force_MI(np.array(z_test), np.array(y_pred>0.5)))
        CMI_test2.append(H.brute_force_CMI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    
        CMI_test_pred2.append(H.brute_force_CMI(np.array(z_test),  np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))
        if i==2:
            fig = H.plot_distributions(y_pred, z_test, fname='Results/MI{}.pdf'.format(gamma))
            fig1 = H.plot_distributions(y_pred[c_test<=dm1], z_test[c_test<=dm1], fname='Results/MI{}_bin1.pdf'.format(gamma))
            fig2 = H.plot_distributions(y_pred[(c_test>dm1)&(c_test<=dm2)], z_test[(c_test>dm1)&(c_test<=dm2)], fname='Results/MI{}_bin2.pdf'.format(gamma))
            fig3 = H.plot_distributions(y_pred[(c_test>dm2)&(c_test<=dm3)], z_test[(c_test>dm2)&(c_test<=dm3)], fname='Results/MI{}_bin3.pdf'.format(gamma))
            fig4 = H.plot_distributions(y_pred[c_test>dm3], z_test[c_test>dm3], fname='Results/MI{}_bin4.pdf'.format(gamma))
            plt.close('all')

    print('Accuracy:{np.mean(np.array(AUC2))}')
    print('AUC:{np.mean(np.array(Acc2))}')
    print('I(Z;Ytrue): {np.mean(np.array(MI_test2))}')
    print('I(Z;Ypred):{np.mean(np.array(MI_test_pred2))}')
    print('I(Z;Ytrue|Xc): {np.mean(np.array(CMI_test2))}')
    print('I(Z;Ypred|Xc): {np.mean(np.array(CMI_test_pred2))}')





#Loss 3 with multiple regularizers
Reg = [2,4]


for j in range(2):
    gamma=Reg[j]
    AUC3=[]
    Acc3=[]
    MI_test3=[]
    MI_test_pred3=[]
    CMI_test3=[]
    CMI_test_pred3=[]
    print('Gamma:{}'.format(gamma))
    for i in range(monte):
        print('Monte-Carlo Iteration:{}'.format(i))
        reset_weights(classifier)
        classifier.compile(optimizer = 'adam', loss = bias3(InputZ,InputC,gamma), metrics = ['accuracy'])
        classifier.fit([X_train, z_train, c_train], y_train, batch_size = 1000, epochs = Epochs,verbose=0)
    
        # predict on test set
        y_pred = classifier.predict([X_test, z_test, c_test]).ravel()
        AUC3.append(roc_auc_score(y_test, y_pred))
        Acc3.append(100*accuracy_score(y_test, (y_pred>0.5)))
        MI_test3.append(H.brute_force_MI(np.array(z_test), np.array(y_test)))
        MI_test_pred3.append(H.brute_force_MI(np.array(z_test), np.array(y_pred>0.5)))
        CMI_test3.append(H.brute_force_CMI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    
        CMI_test_pred3.append(H.brute_force_CMI(np.array(z_test), np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))
   
        if i==2:
            fig = H.plot_distributions(y_pred, z_test, fname='Results/CMI{}.pdf'.format(gamma))
            fig1 = H.plot_distributions(y_pred[c_test<=dm1], z_test[c_test<=dm1], fname='Results/CMI{}_bin1.pdf'.format(gamma))
            fig2 = H.plot_distributions(y_pred[(c_test>dm1)&(c_test<=dm2)], z_test[(c_test>dm1)&(c_test<=dm2)], fname='Results/CMI{}_bin2.pdf'.format(gamma))
            fig3 = H.plot_distributions(y_pred[(c_test>dm2)&(c_test<=dm3)], z_test[(c_test>dm2)&(c_test<=dm3)], fname='Results/CMI{}_bin3.pdf'.format(gamma))
            fig4 = H.plot_distributions(y_pred[c_test>dm3], z_test[c_test>dm3], fname='Results/CMI{}_bin4.pdf'.format(gamma))
    
            plt.close('all')



    print('Accuracy:{np.mean(np.array(AUC3))}')
    print('AUC:{np.mean(np.array(Acc3))}')
    print('I(Z;Ytrue): {np.mean(np.array(MI_test3))}')
    print('I(Z;Ypred):{np.mean(np.array(MI_test_pred3))}')
    print('I(Z;Ytrue|Xc): {np.mean(np.array(CMI_test3))}')
    print('I(Z;Ypred|Xc): {np.mean(np.array(CMI_test_pred3))}')
###############################################
