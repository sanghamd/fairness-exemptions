##**Experiments on Adult Dataset with Feature Exemptions**##


import time
import numpy as np
import random
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")
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
monte1=10
f = open('Results/Results_No_Fairness.txt','w')


#Loading the Dataset
X, y, z, c = H.load_ICU_data('adult.data')

z = np.array(z)
c = np.array(c)
y = np.array(y)


#Defining the bins for the critical feature hours per week
dm1=39
dm2=40
dm3=45
print(dm1)
print(dm2)
print(dm3)
#Roughly at the Quantiles


n_features=X.shape[1]

#Defining the ML model
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




# Create a loss function for maximum accuracy with no fairness
def bias1(lz,lx):
    def loss1(y_true,y_pred):
        Z = lz
        L1 = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        return L1
    return loss1




# Repeat multiple times for each loss function for average case results
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


#Loss1: No Fairness
AUC1=[]
Acc1=[]
MI_test1=[]
MI_test_pred1=[]
CMI_test1=[]
CMI_test_pred1=[]
UNI_test1=[]
UNI_test_pred1=[]

for i in range(monte1):
    print('Monte-Carlo Iteration:{}'.format(i))
    reset_weights(classifier)
    X_train, X_test, y_train, y_test, z_train, z_test, c_train, c_test = train_test_split(X, y, z, c, test_size = 0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.compile(optimizer = 'adam', loss = bias1(InputZ,InputC), metrics = ['accuracy'])
    classifier.fit([X_train, z_train, c_train], y_train, batch_size = 1000, epochs = 20,verbose=1)

    # predict on test set
    y_pred = classifier.predict([X_test, z_test, c_test]).ravel()
    AUC1.append(roc_auc_score(y_test, y_pred))
    Acc1.append(100*accuracy_score(y_test, (y_pred>0.5)))
    MI_test1.append(H.brute_force_MI(np.array(z_test), np.array(y_test)))
    MI_test_pred1.append(H.brute_force_MI(np.array(z_test), np.array(y_pred>0.5)))
    CMI_test1.append(H.brute_force_CMI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    CMI_test_pred1.append(H.brute_force_CMI(np.array(z_test), np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))

    UNI_test1.append(H.UNI(np.array(z_test), np.array(y_test),np.array(c_test),dm1,dm2,dm3))
    UNI_test_pred1.append(H.UNI(np.array(z_test), np.array(y_pred>0.5),np.array(c_test),dm1,dm2,dm3))
           
    # plotting histograms for one monte-carlo case
    if i==2:
        fig = H.plot_distributions(y_pred, z_test, fname='Results/no_fairness.pdf')
        fig1 = H.plot_distributions(y_pred[c_test<=dm1], z_test[c_test<=dm1], fname='Results/no_fairness_bin1.pdf')
        fig2 = H.plot_distributions(y_pred[(c_test>dm1)&(c_test<=dm2)], z_test[(c_test>dm1)&(c_test<=dm2)], fname='Results/no_fairness_bin2.pdf')
        fig3 = H.plot_distributions(y_pred[(c_test>dm2)&(c_test<=dm3)], z_test[(c_test>dm2)&(c_test<=dm3)], fname='Results/no_fairness_bin3.pdf')
        fig4 = H.plot_distributions(y_pred[c_test>dm3], z_test[c_test>dm3], fname='Results/no_fairness_bin4.pdf')

    plt.close('all')



f.write('AUC:{}\n'.format(np.mean(np.array(AUC1))))
f.write('AUC(Median):{}({})\n'.format(np.median(np.array(AUC1)),np.std(np.array(AUC1))))
f.write('Accuracy:{}\n'.format(np.mean(np.array(Acc1))))
f.write('Accuracy(Median):{}({})\n'.format(np.median(np.array(Acc1)),np.std(np.array(Acc1))))
f.write('MI with True Labels:{}\n'.format(np.mean(np.array(MI_test1))))
f.write('MI with Output:{}\n'.format(np.mean(np.array(MI_test_pred1))))
f.write('CMI with True Labels:{}\n'.format(np.mean(np.array(CMI_test1))))
f.write('CMI with Output:{}\n'.format(np.mean(np.array(CMI_test_pred1))))
f.write('UNI with True Labels:{}\n'.format(np.mean(np.array(UNI_test1))))
f.write('UNI with Output:{}\n'.format(np.mean(np.array(UNI_test_pred1))))





f.close()
###############################################
