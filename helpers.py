##Helper File containing additional functions for main file##

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

font = {'weight' : 'bold'}

rc('font', **font)

rc('legend', fontsize=28)
rc('xtick', labelsize=28)


EPS = np.finfo(float).eps


#Function to Print Histograms
def plot_distributions(y, z,fname=None):
    fig, axes = plt.subplots(figsize=(10, 4),sharey=True)
    legend={'gender': ['Z=0','Z=1']}
    attr='gender'
    for attr_val in [0, 1]:
        ax = sns.distplot(y[z == attr_val], hist=False,kde_kws={'shade': True,},label='{}'.format(legend[attr][attr_val]),ax=axes)
    ax.set_xlim(0,1)
    ax.set_ylim(0,8)
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    return fig



# Function to Read the Data
def load_ICU_data(path):
    input_data = (pd.read_csv(path,na_values="?", sep=r'\s*,\s*',engine= 'python').loc[:,:])
    
    # select sensitive attribute
    z = (input_data['gender'] == 'Male').astype(int)
    
    # select true label
    y = (input_data['income'] == '>50K').astype(int)
    
    # select critical feature
    c = (input_data['hours-per-week']).astype(float)
                    
    # select features ('income' and sentive attribute columns are dropped)
    X = (input_data.drop(columns=['income', 'gender']).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
                    
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    return X, y, z, c



#Correlation-Based MI
def MI(a, b):
    ma = K.mean(a)
    mb = K.mean(b)
    am, bm = a-ma, b-mb
    r_num = K.square(K.sum(am * bm))
    r_den = (K.sum(K.square(am)) * K.sum(K.square(bm))) +EPS
    r = r_num / r_den
    return -K.log(1-r+EPS)


#Correlation-Based CMI
def CMI(a,b,d,dm1,dm2,dm3):
    mask1=K.less_equal(d,dm1)
    mask2=K.all(K.stack([K.less_equal(d,dm2), K.greater(d,dm1)], axis=0), axis=0)
    mask3=K.all(K.stack([K.less_equal(d,dm3), K.greater(d,dm2)], axis=0), axis=0)
    mask4=K.greater(d,dm3)
    m1=MI(tf.boolean_mask(a,mask1),tf.boolean_mask(b,mask1))
    m2=MI(tf.boolean_mask(a,mask2),tf.boolean_mask(b,mask2))
    m3=MI(tf.boolean_mask(a,mask3),tf.boolean_mask(b,mask3))
    m4=MI(tf.boolean_mask(a,mask4),tf.boolean_mask(b,mask4))
    p1=K.sum(K.cast(mask1,'float32'))
    p2=K.sum(K.cast(mask2,'float32'))
    p3=K.sum(K.cast(mask3,'float32'))
    p4=K.sum(K.cast(mask4,'float32'))
    return tf.divide(p1,p1+p2+p3+p4)*m1 + tf.divide(p2,p1+p2+p3+p4)*m2 + tf.divide(p3,p1+p2+p3+p4)*m3 +tf.divide(p4,p1+p2+p3+p4)*m4


# Brute Force Computation of MI/CMI in the Actual Dataset for Evaluation


def trunc_log2(a,b,c):
    if a==0:
        return 0
    else:
        return a*np.log2(a/((a+b)*(a+c)))


def brute_force_MI(a,b):
    A=np.array([a,b])
    A=np.transpose(A)
    den=len(A)
    
    pa=(A == [0,0]).all(-1).sum()/den
    pb=(A == [1,0]).all(-1).sum()/den
    pc=(A == [0,1]).all(-1).sum()/den
    pd=(A == [1,1]).all(-1).sum()/den
    
    mi=trunc_log2(pa,pb,pc) + trunc_log2(pb,pd,pa) + trunc_log2(pc,pa,pd) + trunc_log2(pd,pb,pc)
    return mi



def brute_force_CMI(a,b,c,dm1,dm2,dm3):
    m1=brute_force_MI(a[c<=dm1],b[c<=dm1])
    p1=len(a[c<=dm1])
    m2=brute_force_MI(a[(c>dm1)&(c<=dm2)],b[(c>dm1)&(c<=dm2)])
    p2=len(a[(c>dm1)&(c<=dm2)])
    m3=brute_force_MI(a[(c>dm2)&(c<=dm3)],b[(c>dm2)&(c<=dm3)])
    p3=len(a[(c>dm2)&(c<=dm3)])
    m4=brute_force_MI(a[c>dm3],b[c>dm3])
    p4=len(a[c>dm3])
    cmi= (p1/(p1+p2+p3+p4))*m1+ (p2/(p1+p2+p3+p4))*m2+ (p3/(p1+p2+p3+p4))*m3+ (p4/(p1+p2+p3+p4))*m4
             
    return cmi



