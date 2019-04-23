"""Constants declared and assigned which are imported to main file.

.. moduleauthor:: Riccardo Affolter <riccardo.affolter@gmail.com>
"""
from pprint import pprint
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from datamodel.DataModel import Tab,Img

class BlackBox:
    def start(self, datamodel):
        print("Model not present")
        if isinstance(datamodel, Tab):
            cls = self.bb
            param = self.param_grid
            if (datamodel.e):
                ann = RandomizedSearchCV(estimator=cls, param_distributions=param, n_iter=5, cv=3, scoring='accuracy',
                                         random_state=datamodel.seed, n_jobs=-1)
            else:
                ann = GridSearchCV(estimator=cls, param_grid=param, cv=2, scoring="accuracy", n_jobs=4,verbose = 2)
            scv = ann.fit(datamodel.train_features, datamodel.train_labels)
            print("Best parameters set found on development set:")
            print()
            print(scv.best_params_)
            print()
            print("Grid scores on training set:")
            print()
            means = scv.cv_results_['mean_test_score']
            stds = scv.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, scv.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            rf = scv.best_estimator_
            datamodel.set_cls(rf)
            pprint(datamodel.getCls().get_params())
            datamodel.saveModel(str(datamodel.filename.split(".")[0] + '_' + datamodel.name + '.pickle'))
        elif isinstance(datamodel, Img):
            bb = BlackBox.CNN(datamodel)
            bb.cls.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])
            bb.fit(datamodel)
            img = bb.start(datamodel)
            model_json = img.cls.to_json()
        return datamodel

class NeuralN(BlackBox):

    def __init__(self,seed):
        self.param_grid = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [(100,), (3, 3, 3), (5, 2)]}
        self.bb = MLPClassifier(random_state=seed)

class Svm(BlackBox):

    def __init__(self, seed):
        Cs = [0.001, 0.1, 1, 10]
        gammas = [0.001, 0.1, 1]
        self.param_grid = {'C': Cs, 'gamma': gammas}
        self.bb = SVC(kernel='rbf',probability=True, random_state=seed)

class RandomForest(BlackBox):

    def __init__(self, seed):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        self.param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        self.bb = RandomForestClassifier(random_state=seed)

class CNN(BlackBox):

    #function that uses keras.evaluate
    def evaluate(self,model):
        score = model.cls.evaluate(model.x_test, model.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def __init__(self,model):
        num_classes = 10
        epochs = 3
        batch_size = 128
        self.cls = Sequential()
        self.cls.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=model.input_shape))
        self.cls.add(Conv2D(64, (3, 3), activation='relu'))
        self.cls.add(MaxPooling2D(pool_size=(2, 2)))
        self.cls.add(Dropout(0.25))
        self.cls.add(Flatten())
        self.cls.add(Dense(128, activation='relu'))
        self.cls.add(Dropout(0.5))
        self.cls.add(Dense(num_classes))
        self.cls.add(Activation('softmax'))
        self.cls.fit(model.x_train, model.y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(model.x_test, model.y_test))


