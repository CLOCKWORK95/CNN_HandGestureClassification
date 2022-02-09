
from tabnanny import verbose
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import KFold, StratifiedKFold
from utilities import csv_readnstack, csv_to_onehot, scalings
from models import BigJohn, LittleJohn, tuning_model
import pandas as pd
import numpy as np

def main() :

    dataset_values = csv_readnstack( [ "train_gesture_x.csv", "train_gesture_y.csv", "train_gesture_z.csv" ] )
    dataset_labels = csv_to_onehot( "train_label.csv" )
    n_classes = 8

    # Split del Dataset in Training Set e Test Set
    train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split( dataset_values, 
                                                                                            dataset_labels, 
                                                                                            test_size = .2, 
                                                                                            shuffle = True )
    # Input Shape settings ( dimensione vettore, numero di canali )
    input_shape = ( train_set_values.shape[1], train_set_values.shape[2] )


    # Split del Training Set in Training Set e Validation Set
    train_set_values, val_set_values, train_set_labels, val_set_labels = train_test_split(  train_set_values, 
                                                                                            train_set_labels, 
                                                                                            test_size = .2, 
                                                                                            shuffle = True )          
    # Scaling dei dati di input secondo la regola Robust Scaling
    datasets = [ train_set_values, val_set_values, test_set_values ]
    train_set_values, val_set_values, test_set_values = scalings( datasets )

    # Creazione e Compilazione del Modello CNN
    model = LittleJohn( input_shape )

    # Setting per l'Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss', patience = 30, verbose = 1,
        mode = 'auto', baseline = None, restore_best_weights = True
    )

    # Addestramento
    history = model.fit(train_set_values, train_set_labels, validation_data = (val_set_values,val_set_labels),
                        batch_size = 256, epochs = 300, verbose = 1, callbacks = [callback] )

    # Valutazione del Modello sull'insieme di Test
    performance = model.evaluate( test_set_values, test_set_labels, verbose = 0 )
    print( "Validation performance" )
    print( performance )

    # Plot dei risultati sulla loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



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
        indexes = dflabels.index[ dflabels.iloc[:,-1] == gesture[0] ].tolist()
        for i in indexes :
            newlabel = [ gesture[1] ], newlabels.append( newlabel )

            newrowx = dfxvalues[i,:].reverse()
            newdataset_x.append( newrowx )

            newrowy = dfyvalues[i,:].reverse()
            newdataset_y.append( newrowy )

            newrowz = dfzvalues[i,:].reverse()
            newdataset_z.append( newrowz )
    
    newdataset = [ newdataset_x, newdataset_y, newdataset_z]
    newdataset = np.dstack( newdataset )
    newlabels = tf.keras.utils.to_categorical( newlabels, num_classes = 8 )

    print( newdataset.shape )
    print( newlabels.shape )
    
    return newdataset, newlabels






if __name__ == '__main__':
    #main()
    data_augmentation()