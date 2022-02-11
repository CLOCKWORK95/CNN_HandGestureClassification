
from sklearn.model_selection import StratifiedKFold,KFold
from utilities import csv_readnstack, csv_to_onehot, data_augmentation
from models import BigJohn, BigJohnTuned, LittleJohn, cvBigJohn
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading
import time

EPOCHS = 3
K = 3

# K-Fold Cross-Validation variables
TRAINERS = []
LOSSES1 = [ 0.0 for i in range( 0, K ) ]
ACCURACIES1 = [ 0.0 for i in range( 0, K ) ]
LOSSES2 = [ 0.0 for i in range( 0, K ) ]
ACCURACIES2 = [ 0.0 for i in range( 0, K ) ]
NAME1 = ""
NAME2 = ""


class K_trainer ( threading.Thread ):

    def __init__( self, index, id, input_shape, dataset_values, dataset_labels, train_index, val_index ):
        threading.Thread.__init__(self)
        self.index = index        
        self.id = id
        self.input_shape = input_shape
        self.dataset_values = dataset_values
        self.dataset_labels = dataset_labels
        self.train_index = train_index
        self.val_index = val_index

    def run( self ):

        global LOSSES1, LOSSES2, ACCURACIES1, ACCURACIES2, NAME1, NAME2

        training_values, training_labels = self.dataset_values[ self.train_index ], self.dataset_labels[ self.train_index ] 
        validation_values, validation_labels = self.dataset_values[ self.val_index ], self.dataset_labels[ self.val_index ]
        
        callback = tf.keras.callbacks.EarlyStopping(    monitor = 'loss', patience = 25, verbose = 1, 
                                                        mode = 'auto', baseline = None, restore_best_weights = True ) 
        
        model1, NAME1 = LittleJohn( self.input_shape ) 
        model2, NAME2 = BigJohn( self.input_shape )

        model1.fit( training_values, training_labels, epochs = EPOCHS, batch_size = 256, verbose = 1, callbacks = [ callback ] )
        val_loss1, val_accuracy1 = model1.evaluate( validation_values, validation_labels )

        LOSSES1[self.index] = val_loss1 
        ACCURACIES1[self.index] =  val_accuracy1 

        model2.fit( training_values, training_labels, epochs = EPOCHS, batch_size = 256, verbose = 1, callbacks = [ callback ] )
        val_loss2, val_accuracy2 = model2.evaluate( validation_values, validation_labels )

        LOSSES2[self.index] = val_loss2 
        ACCURACIES2[self.index] =  val_accuracy2 

        model1.save("serialized/KFOLDS_CROSSVAL/KFCV_" + str(self.id) + "_" + NAME1 + "_" + NAME2 + "/K_"+ NAME1 + str(self.index) + "_" + str( val_accuracy1 ) + ".h5", True, True)
        model1.save("serialized/KFOLDS_CROSSVAL/KFCV_" + str(self.id) + "_" + NAME1 + "_" + NAME2 + "/K_"+ NAME2 + str(self.index) + "_" + str( val_accuracy2 ) + ".h5", True, True)

        return


def kFold_Validation( input_shape, dataset_values, labels, dataset_labels, K ) :
    # K-Fold Validation
    global EPOCHS

    id = 0
    with open( "serialized/KFOLDS/id.txt", mode = 'r' ) as f :
        id = int( f.read() )
        id = id + 1
        f.close()
    with open( "serialized/KFOLDS/id.txt", mode = 'w+' ) as f :
        f.write( str(id) )
        f.close()
    
    loss_array = []
    accuracy_array = []

    counter = 1
    for train_index, val_index in StratifiedKFold(K, shuffle=True).split( dataset_values, labels ): 
        # Split del TS in K parti
        print( len(train_index), len(val_index) )
        print("\n\n")

        training_values, training_labels = dataset_values[ train_index ], dataset_labels[ train_index ] 
        validation_values, validation_labels = dataset_values[ val_index ], dataset_labels[ val_index ]
        
        callback = tf.keras.callbacks.EarlyStopping(    monitor = 'loss', patience = 25, verbose = 1, 
                                                        mode = 'auto', baseline = None, restore_best_weights = True ) 
        
        model1, name1 = LittleJohn( input_shape ) 

        model1.fit( training_values, training_labels, epochs = EPOCHS, batch_size = 256, verbose = 1, callbacks = [ callback ] )

        val_loss, val_accuracy = model1.evaluate( validation_values, validation_labels )

        loss_array.append( val_loss )
        accuracy_array.append( val_accuracy )

        print("\nModel Evaluation " + str(counter) + " - " + name1 + " - validation loss :", val_loss )
        print("\nModel Evaluation " + str(counter) + " - " + name1 + " - validation accuracy :", val_accuracy )

        model1.save("serialized/KFOLDS/KFOLD" + str(id) + "_" + name1 + "/K_" + str(counter) + "_" + str( val_accuracy ) + ".h5", True, True)

        counter = counter + 1
    
    loss_array = np.array( loss_array )
    accuracy_array = np.array( accuracy_array )
    meanloss = np.mean(loss_array)
    meanaccuracy = np.mean(accuracy_array)

    with open( "serialized/KFOLDS/KFOLD" + str(id) + "_" + name1 + "/report.txt", 'w' ) as f :
        f.write( "Mean Accuracy : " + str(meanaccuracy) + "\n" + "Mean Loss : " + str(meanloss) )
        f.close()


def KFold_Cross_Validation( input_shape, dataset_values, labels, dataset_labels, K ) :
    # K-Fold Cross-Validation
    global EPOCHS, TRAINERS, LOSSES1, LOSSES2, ACCURACIES1, ACCURACIES2, NAME1, NAME2

    id = 0
    with open( "serialized/KFOLDS_CROSSVAL/id.txt", mode = 'r' ) as f :
        id = int( f.read() )
        id = id + 1
        f.close()
    with open( "serialized/KFOLDS_CROSSVAL/id.txt", mode = 'w+' ) as f :
        f.write( str(id) )
        f.close()

    skf =  StratifiedKFold( K, shuffle = True ).split( dataset_values, labels ) 

    counter = 0
    for train_index, val_index in skf :
        trainer_thread = K_trainer( counter, id, input_shape, dataset_values, dataset_labels, train_index, val_index )
        TRAINERS.append( trainer_thread )
        trainer_thread.start()
        counter = counter + 1
    for trainer in TRAINERS :
        trainer.join()
    
    LOSSES1 = np.array( LOSSES1 )
    ACCURACIES1 = np.array( ACCURACIES1 )
    LOSSES2 = np.array( LOSSES2 )
    ACCURACIES2 = np.array( ACCURACIES2 )
    print(ACCURACIES1)

    meanloss1, meanloss2 = np.mean(LOSSES1), np.mean(LOSSES2)
    meanaccuracy1, meanaccuracy2 = np.mean(ACCURACIES1), np.mean(ACCURACIES2)

    with open( "serialized/KFOLDS_CROSSVAL/KFCV_" + str(id) + "_" + NAME1 + "_" + NAME2 + "/cv_report", 'w' ) as f :
        f.write( "Mean Accuracy " + NAME1 + " : " + str(meanaccuracy1) + "\n" + "Mean Loss " + NAME1 + " : " + str(meanloss1)+
               "\nMean Accuracy " + NAME2 + " : " + str(meanaccuracy2) + "\n" + "Mean Loss " + NAME2 + " : " + str(meanloss2) )
        f.close()
    


def main():

    global K

    # Preprocessing dei dati
    dataset_values = csv_readnstack( [ "train_gesture_x.csv", "train_gesture_y.csv", "train_gesture_z.csv" ] )
    dataset_labels = csv_to_onehot( "train_label.csv" )
    labels = pd.read_csv( "train_label.csv", header = None, dtype = 'float64' )
    labels = labels.values

    # Concatazione di nuove entries al dataset ottenute tramite Data Augmentation
    augmentation_values, augmentation_labels, new_labels = data_augmentation()
    dataset_values = np.vstack( (dataset_values, augmentation_values) )
    dataset_labels = np.vstack( (dataset_labels, augmentation_labels) )
    labels = np.concatenate( (labels, new_labels) )

    print( dataset_values.shape )
    print( dataset_labels.shape )
    print( labels.shape )

    # Input Shape settings ( dimensione vettore, numero di canali )
    input_shape = ( dataset_values.shape[1], dataset_values.shape[2] )

    #kFold_Validation( input_shape, dataset_values, labels, dataset_labels, K )

    KFold_Cross_Validation( input_shape, dataset_values, labels, dataset_labels, K )

if __name__ == '__main__':

    main()