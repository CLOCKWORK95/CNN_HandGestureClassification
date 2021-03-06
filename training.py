
from tabnanny import verbose
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import KFold, StratifiedKFold
from utilities import csv_readnstack, csv_to_onehot, scalings, data_augmentation
from models import BigJohn, BigJohnTuned, HyperBigJohn, LittleJohn, JohnnyBoy, tuning_model
import pandas as pd
import numpy as np


def main() :

    dataset_values = csv_readnstack( [ "train_gesture_x.csv", "train_gesture_y.csv", "train_gesture_z.csv" ] )
    dataset_labels = csv_to_onehot( "train_label.csv" )
    n_classes = 8
    
    # Concatazione di nuove entries al dataset ottenute tramite Data Augmentation
    augmentation_values, augmentation_labels, labels = data_augmentation()
    dataset_values = np.vstack( (dataset_values, augmentation_values) )
    dataset_labels = np.vstack( (dataset_labels, augmentation_labels) )
    print( dataset_values.shape )
    print( dataset_labels.shape )
    
    # Split del Dataset in Training Set e Test Set
    train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split( dataset_values, 
                                                                                            dataset_labels, 
                                                                                            test_size = .3, 
                                                                                            shuffle = True )
    # Input Shape settings ( dimensione vettore, numero di canali )
    input_shape = ( train_set_values.shape[1], train_set_values.shape[2] )


    # Split del Training Set in Training Set e Validation Set
    train_set_values, val_set_values, train_set_labels, val_set_labels = train_test_split(  train_set_values, 
                                                                                            train_set_labels, 
                                                                                            test_size = .25, 
                                                                                            shuffle = True )          
    # Scaling dei dati di input secondo la regola Robust Scaling
    datasets = [ train_set_values, val_set_values, test_set_values ]
    train_set_values, val_set_values, test_set_values = scalings( datasets )

    # Creazione e Compilazione del Modello CNN
    model, name = BigJohnTuned( input_shape )

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

    model.save("serialized/model" + name + str( performance[1]) + '.h5' , True, True)
    
    # Plot dei risultati sulla loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()


if __name__ == '__main__':
    main()
    #data_augmentation()