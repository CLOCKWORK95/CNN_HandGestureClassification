
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, HalvingGridSearchCV
from models import tuning_model


def hyperparams_tuning( train_x, train_y ):
    # Tuning degli iperparametri in Cross Validation
    model = KerasClassifier( build_fn = tuning_model, verbose = 1 )
    batch_size = [ 64 ]
    opt = ['adam']
    fsize1 = [ 15 ]
    neurons_1 = [ 128 ]
    epochs = [ 80 ]
    act = ['tanh']

    param_grid = dict(  
                        batch_size = batch_size,
                        epochs = epochs, 
                        opt = opt, 
                        fsize1 = fsize1,
                        act = act,
                        neurons_1 = neurons_1 )

    grid = HalvingGridSearchCV( estimator = model,
                                param_grid = param_grid,
                                n_jobs = -1,
                                cv = 5)

    callback = tf.keras.callbacks.EarlyStopping( monitor = 'accuracy', restore_best_weights = True, patience = 30 )

    grid_result = grid.fit( train_x, train_y, callbacks = [callback] )
    # Stampa del best Score dalla Grid Search
    print( "Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_ ) )
