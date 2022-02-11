
import tensorflow as tf


def tuning_model( neurons, exponent, ksize, numf, act, input_shape ) :
    # Creazione del Modello CNN da tunare ( Iper-Parametri ) [ BigJohn ]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = numf, kernel_size = ksize, activation='relu', padding = "same", input_shape = input_shape, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(   filters = int(numf/exponent), kernel_size = int(ksize/exponent), activation='relu', padding = "same" , 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size = 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(    neurons , activation = act, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    int(neurons/(2**exponent)), activation = act, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.1)) 
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    int(neurons/(2**(exponent+1))), activation = act, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    model.add(tf.keras.layers.Dense( 8 , activation = 'softmax',  kernel_regularizer = tf.keras.regularizers.l2(1.e-4) ))
    
    #model.summary()

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

   

    return model


def cvBigJohn( input_shape ) :

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

    name = 'BigJohn'

    return model, name


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

    name = 'LittleJohn'

    return model, name


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

    name = 'JohnnyBoy'

    return model, name


def BigJohnTuned( input_shape ) :

    # Creazione del Modello CNN

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 15, activation='relu', padding = "same", input_shape = input_shape, 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size= 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(   filters = 64, kernel_size = 10, activation='relu', padding = "same" , 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.AveragePooling1D( pool_size = 8, strides = 5, padding = "same") )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(    512, kernel_initializer = tf.keras.initializers.HeNormal() ,activation='relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.1))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    288, kernel_initializer = tf.keras.initializers.HeNormal() ,activation = 'relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))
    model.add(tf.keras.layers.Dropout(.2)) 
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(    128, kernel_initializer = tf.keras.initializers.HeNormal() ,activation = 'relu', 
                                        kernel_regularizer = tf.keras.regularizers.l2(1.e-4)))

    model.add(tf.keras.layers.Dense( 8 , activation = 'softmax',  kernel_regularizer = tf.keras.regularizers.l2(1.e-4) ))
    
    #model.summary()

    opt = tf.keras.optimizers.Adam()
    model.compile( optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'] )

    name = 'BigJohnTuned'

    return model, name
