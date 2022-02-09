
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, HalvingGridSearchCV
from models import tuning_model
import numpy as np
from utilities import csv_readnstack, csv_to_onehot, data_augmentation


def hyperparams_tuning( train_x, train_y, input_shape ):

    # Tuning degli iperparametri in Cross Validation
    model = KerasClassifier( build_fn = tuning_model, verbose = 1 )
    batch_size = [ 256 ]
    opt = ['adam']
    neurons = [ 256, 512 ]
    exponent = [ 1, 2 ]
    numf = [ 64, 32 ]
    ksize = [ 10 ]
    epochs = [ 80 ]
    act = ['tanh', 'relu']
    input_shape = [ input_shape ]

    param_grid = dict(  
                        batch_size = batch_size,
                        input_shape = input_shape,
                        epochs = epochs, 
                        ksize = ksize,
                        exponent = exponent,
                        numf = numf,
                        act = act,
                        neurons = neurons )

    grid = HalvingGridSearchCV( estimator = model,
                                param_grid = param_grid,
                                n_jobs = -1,
                                cv = 5)

    callback = tf.keras.callbacks.EarlyStopping( monitor = 'accuracy', restore_best_weights = True, patience = 20 )

    grid_result = grid.fit( train_x, train_y, callbacks = [callback] )
    # Stampa del best Score dalla Grid Search
    print( "Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_ ) )



def tune_hyper_parameters():

    dataset_values = csv_readnstack( [ "train_gesture_x.csv", "train_gesture_y.csv", "train_gesture_z.csv" ] )
    dataset_labels = csv_to_onehot( "train_label.csv" )

    # Concatazione di nuove entries al dataset ottenute tramite Data Augmentation
    augmentation_values, augmentation_labels = data_augmentation()
    dataset_values = np.vstack( (dataset_values, augmentation_values) )
    dataset_labels = np.vstack( (dataset_labels, augmentation_labels) )
    print( dataset_values.shape )
    print( dataset_labels.shape )

    # Input Shape settings ( dimensione vettore, numero di canali )
    input_shape = ( dataset_values.shape[1], dataset_values.shape[2] )

    hyperparams_tuning( dataset_values, dataset_labels, input_shape )



if __name__ == '__main__':
    tune_hyper_parameters()
    