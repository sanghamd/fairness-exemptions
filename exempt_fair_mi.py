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


#Define Monte-Carlo Iterations
monte2=10
f = open('Results/Results_MI.txt','w')


#Loading the Dataset
X, y, z, c = H.load_ICU_data('adult.data')

z = np.array(z)
c = np.array(c)
y = np.array(y)


# Defining the bins for the critical feature hours per week
dm1=39
dm2=40
dm3=45
print(dm1)
print(dm2)
print(dm3)
n_features=X.shape[1]

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





# Create a loss function with MI as regularizer
def bias2(lz,lx,gamma):
    def loss2(y_true,y_pred):
        Z = lz
        L1 = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        L2 = H.MI(Z,y_pred)
        return L1 + gamma*L2
    return loss2



# Repeat multiple times for each loss function for average case results
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


#Loss 2 with multiple regularizers
Reg=[0.5,1,2,4,10,20,50,100]
#Reg=[4]
n_Reg=len(Reg)

for j in range(n_Reg):
    gamma = Reg[j]
    AUC2=[]
    Acc2=[]
    MI_test2=[]
    MI_test_pred2=[]
    CMI_test2=[]
    CMI_test_pred2=[]
    UNI_test2=[]
    UNI_test_pred2=[]
    f.write('Gamma:{}\n'.format(gamma))
    for i in range(monte2):
        print('Monte-Carlo Iteration:{}'.format(i))
        reset_weights(classifier)
        X_train, X_test, y_train, y_test, z_train, z_test, c_train, c_test = train_test_split(X, y, z, c, test_size = 0.3)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier.compile(optimizer = 'adam', loss = bias2(InputZ,InputC,gamma), metrics = ['accuracy'])
        classifier.fit([X_train, z_train, c_train], y_train, batch_size = 1000, epochs = 50,verbose=1)
    
        # predict on test set
        y_pred = classifier.predict([X_test, z_test, c_test]).ravel()
        AUC2.append(roc_auc_score(y_test, y_pred))
        Acc2.append(100*accuracy_score(y_test, (y_pred>0.5)))
        MI_test2.append(H.brute_force_MI(np.array(z_test),np.array(y_test)))
        MI_test_pred2.append(H.brute_force_MI(np.array(z_test), np.array(y_pred>0.5)))
        CMI_test2.append(H.brute_force_CMI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    
        CMI_test_pred2.append(H.brute_force_CMI(np.array(z_test),  np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))
        UNI_test2.append(H.UNI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
        UNI_test_pred2.append(H.UNI(np.array(z_test), np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))


    
        if i==2:
            fig = H.plot_distributions(y_pred, z_test, fname='Results/MI{}.pdf'.format(gamma))
            fig1 = H.plot_distributions(y_pred[c_test<=dm1], z_test[c_test<=dm1], fname='Results/MI{}_bin1.pdf'.format(gamma))
            fig2 = H.plot_distributions(y_pred[(c_test>dm1)&(c_test<=dm2)], z_test[(c_test>dm1)&(c_test<=dm2)], fname='Results/MI{}_bin2.pdf'.format(gamma))
            fig3 = H.plot_distributions(y_pred[(c_test>dm2)&(c_test<=dm3)], z_test[(c_test>dm2)&(c_test<=dm3)], fname='Results/MI{}_bin3.pdf'.format(gamma))
            fig4 = H.plot_distributions(y_pred[c_test>dm3], z_test[c_test>dm3], fname='Results/MI{}_bin4.pdf'.format(gamma))
        plt.close('all')

    f.write('AUC:{}\n'.format(np.mean(np.array(AUC2))))
    f.write('AUC(Median):{}({})\n'.format(np.median(np.array(AUC2)),np.std(np.array(AUC2))))
    f.write('Accuracy:{}\n'.format(np.mean(np.array(Acc2))))
    f.write('Accuracy(Median):{}({})\n'.format(np.median(np.array(Acc2)),np.std(np.array(Acc2))))
    f.write('MI with True Labels:{}\n'.format(np.mean(np.array(MI_test2))))
    f.write('MI with Output:{}\n'.format(np.mean(np.array(MI_test_pred2))))
    f.write('CMI with True Labels:{}\n'.format(np.mean(np.array(CMI_test2))))
    f.write('CMI with Output:{}\n'.format(np.mean(np.array(CMI_test_pred2))))
    f.write('UNI with True Labels:{}\n'.format(np.mean(np.array(UNI_test2))))
    f.write('UNI with Output:{}\n'.format(np.mean(np.array(UNI_test_pred2))))



f.close()
###############################################
