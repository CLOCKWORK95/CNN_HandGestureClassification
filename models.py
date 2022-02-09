
import tensorflow as tf


def tuning_model( opt, fsize1, neurons_1, act, input_shape ) :
    # Creazione del Modello CNN da tunare ( Iper-Parametri )
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters = 128, kernel_size = fsize1, activation='relu', padding="same", input_shape = input_shape,
                    kernel_regularizer = tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 2 , padding="same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 5,  activation='relu', padding="same", input_shape = input_shape))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 2 , padding="same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 5,  activation='relu', padding="same", input_shape = input_shape))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 2 , padding="same") )
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense( neurons_1, activation = act ))
    model.add(tf.keras.layers.Dropout(.45))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense( neurons_1/2, activation = act ))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense( neurons_1/4, activation = act ))
    model.add(tf.keras.layers.Dense( 8 , activation = 'softmax' ))
    
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    return model


def BigJohn( input_shape ) :

    # Creazione del Modello CNN

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 15, activation='relu', padding = "same", input_shape = input_shape, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(   filters = 32, kernel_size = 10, activation='relu', padding = "same" , 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size = 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(    512, kernel_initializer = tf.keras.initializers.HeNormal() ,activation='relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    128, kernel_initializer = tf.keras.initializers.HeNormal() ,activation = 'relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.1)) 
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    64, kernel_initializer = tf.keras.initializers.HeNormal() ,activation = 'relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    model.add(tf.keras.layers.Dense( 8 , activation = 'softmax',  kernel_regularizer = tf.keras.regularizers.l2(1.e-4) ))
    
    #model.summary()

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    return model


def HyperBigJohn( input_shape ) :

    # Creazione del Modello CNN

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 15, activation='relu', padding = "same", input_shape = input_shape, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(   filters = 32, kernel_size = 10, activation='relu', padding = "same" , 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size = 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(    512, kernel_initializer = tf.keras.initializers.GlorotNormal() ,activation='tanh', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    128, kernel_initializer = tf.keras.initializers.GlorotNormal() ,activation = 'tanh', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.15)) 
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    64, kernel_initializer = tf.keras.initializers.GlorotNormal() ,activation = 'tanh', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    model.add(tf.keras.layers.Dense( 8 , activation = 'softmax',  kernel_regularizer = tf.keras.regularizers.l2(1.e-4) ))
    
    #model.summary()

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    return model

def LittleJohn( input_shape ) :

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 10, activation='relu', 
                                        kernel_initializer = tf.keras.initializers.HeNormal(), 
                                        padding = "same", input_shape = input_shape,
                                        kernel_regularizer = tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 10,  activation='relu', 
                                        padding = "same"))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 2,  activation='relu',  
                                        padding = "same"))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(    512, activation = 'relu',
                                        kernel_initializer = tf.keras.initializers.HeNormal() ))
    model.add(tf.keras.layers.Dropout(.45))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(    128, activation = 'relu', 
                                        kernel_initializer = tf.keras.initializers.HeNormal() ))
    model.add(tf.keras.layers.Dense(    8 , activation = 'softmax',
                                        kernel_regularizer = tf.keras.regularizers.l2(0.001) ))

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    return model


def JohnnyBoy( input_shape ) :

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 10, activation='relu', 
                                        kernel_initializer = tf.keras.initializers.HeNormal(), 
                                        padding = "same", input_shape = input_shape,
                                        kernel_regularizer = tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 10,  activation='relu', 
                                        padding = "same"))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 2,  activation='relu',  
                                        padding = "same"))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5 , padding = "same") )

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(    512, activation = 'tanh',
                                        kernel_initializer = tf.keras.initializers.GlorotNormal() ))
    model.add(tf.keras.layers.Dropout(.45))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(    128, activation = 'tanh', 
                                        kernel_initializer = tf.keras.initializers.GlorotNormal() ))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(    8 , activation = 'softmax',
                                        kernel_regularizer = tf.keras.regularizers.l2(0.001) ))

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    return model