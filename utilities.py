
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler


def csv_readnstack( paths ) :
    # @params : [ csv files pathnames ]
    dataset = []
    count = 0
    for path in paths :
        df = pd.read_csv( path )
        df.fillna( method='ffill', axis='index' )
        dataset.append( df.values )
        count = count + 1
    dataset = np.dstack( dataset )
    return dataset

def csv_to_onehot( path ) :
    # @params : csv file pathname
    labels = pd.read_csv( path )
    labels = labels.values
    n_classes = len( np.unique(labels) )
    labels = tf.keras.utils.to_categorical( labels, num_classes = n_classes )
    return labels

def scaling( dataset ):
    # Scaling dei dati secondo tecninca Robust Scaling
    scalers = {}
    for i in range(dataset.shape[1]):
        scalers[i] = RobustScaler()   
        dataset[:, i, :] = scalers[i].fit_transform(dataset[:, i, :]) 
    return dataset

def scalings( datasets ):
    # Scaling dei dati secondo tecninca Robust Scaling
    counter = 0
    for set in datasets :
        datasets[counter] = scaling( set )
        counter = counter + 1
    return ( datasets[counter] for counter in range(0, len(datasets)) )