
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
        df = pd.read_csv( path, header = None, dtype = 'float64' )
        df.fillna( method='ffill', axis='index' )
        dataset.append( df.values )
        count = count + 1
    dataset = np.dstack( dataset )
    return dataset

def csv_to_onehot( path ) :
    # @params : csv file pathname
    labels = pd.read_csv( path, header = None, dtype = 'float64' )
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


def data_augmentation():

    dfxvalues = pd.read_csv( 'train_gesture_x.csv', header = None, dtype = 'float64' )
    dfyvalues = pd.read_csv( 'train_gesture_y.csv', header = None, dtype = 'float64' )
    dfzvalues = pd.read_csv( 'train_gesture_z.csv', header = None, dtype = 'float64' )
    dflabels  = pd.read_csv( 'train_label.csv', header = None, dtype = 'float64' )

    inversion_grid = {  'left' :            [ 3.0, 2.0 ],
                        'right' :           [ 2.0, 3.0 ],
                        'up' :              [ 4.0, 5.0 ],
                        'down' :            [ 5.0, 4.0 ],
                        'circle left' :     [ 7.0, 6.0 ],
                        'circle right' :    [ 6.0, 7.0 ] }

    newdataset_x = []
    newdataset_y = []
    newdataset_z = []
    newlabels = []

    for gesture in inversion_grid :
        indexes = dflabels.index[ dflabels.iloc[:,-1] == inversion_grid[gesture][0] ].tolist()

        for i in indexes :
            newlabel = [ inversion_grid[gesture][1] ]
            newlabels.append( newlabel )

            newrowx = list( reversed( dfxvalues.iloc[i,:].tolist() ) )
            newdataset_x.append( newrowx )

            newrowy = list( reversed( dfyvalues.iloc[i,:].tolist() ) )
            newdataset_y.append( newrowy )

            newrowz = list( reversed( dfzvalues.iloc[i,:].tolist() ) )
            newdataset_z.append( newrowz )
    
    newdataset = [ newdataset_x, newdataset_y, newdataset_z]
    newdataset = np.dstack( newdataset )
    newlabels = tf.keras.utils.to_categorical( newlabels, num_classes = 8 )

    #print( newdataset)
    #print( newlabels )
    
    return newdataset, newlabels

