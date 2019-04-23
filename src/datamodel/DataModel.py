from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
from LOREM.code.util import record2str
import numpy as np
from pprint import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from skimage.color import rgb2gray
from utility.Utility import plot_confusion_matrix
import pandas as pd

class Model:

    def __getattr__(self, attr):
        print("Value {} non present in the model.".format(attr))
        return None

    def get_cls(self):
        return self.cls

    def set_cls(self, x):
        self.cls = x

class Tab(Model):

    @property
    def name_features(self):
        a = pd.DataFrame(data=self.features_names,
                         columns=["Name"])
        return a

    @property
    def bb_params(self):
        pprint(self.cls.get_params())

    @property
    def view_available_record(self):
        a = pd.DataFrame(data=self.test_features,
                         columns=self.features_names)
        a[self.class_name] = self.test_labels
        return a

    def definition(self,df,rdf, *args ,target = "class"):
        np.random.seed(1)
        self.label_encoder = dict()
        self.data = df.values
        self.df = df
        self.rdf = rdf
        self.seed = 1
        self.features_names, self.class_values , self.numeric_columns, self.real_feature_names, self.e = [x for x in args]
        for counter, value in enumerate(self.class_values):
            self.label_encoder.update({counter : value})
        self.categorical_features = list()
        self.categorical_names = {}
        for key, val in self.e.items():
            if len(val) > 1:
                self.categorical_features.append(key)
                self.categorical_names[key] = val
        self.class_name = target

    def saveModel(self, filename):
        pickle.dump(self.cls, open(filename, 'wb'))

    def getModelFromFile(self, filename):
        with open(filename, 'rb') as f:
            self.cls = pickle.load(f)
        return self.cls

    def __featureValues(self):
        return self.df.loc[:, self.df.columns != self.class_name].values

    def getValues(self,df,name):
        df.columns.get_loc(name)

    def getUniqueValue(self,name = "class"):
        return self.df[name].unique()

    def setSplit(self,X,Y,train_size=0.70):
        self.X = X
        self.Y = Y
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(X, Y,
                                                                                                        train_size=train_size,
                                                                                                        random_state=self.seed,
                                                                                                        stratify=self.Y)
    def __set_features_names(self):
        self.feature_names = [obj for i, obj in enumerate(self.df.columns) if i < len(self.df.columns) - 1]

    def __set_class_names(self):
        self.class_names = [obj for i, obj in enumerate(self.df.columns) if i == len(self.df.columns) - 1]

    def set_cls(self, x):
        if(isinstance(x,MLPClassifier)):
            self.bb_name = "Neural Network"
        elif(isinstance(x,SVC)):
            self.bb_name = "Support Vector Machine"
        else:
            self.bb_name ="Random Forest"
        self.cls = x

    def __choose_features(self, n_features):
        return self.features_names[n_features]

    """row = number of roow explained
    prediction = target predicted """
    def toPredict(self,row, prediction):
        print('Prediction(x) = { %s }' % prediction)
        print('x = %s' % record2str(row, self.features_names, self.numeric_columns))
        print('')

    def f1_score(self):
        if(len(self.getUniqueValue()) < 3):
            preds = self.cls.predict(self.test_features)
            f1 = f1_score(self.test_labels, preds)
            print("F1 for {0}: {1}".format(self.class_name, f1))
        else:
            #print("average='binary")
            return

    def evaluate(self):
        print("BlackBox: {}".format(self.bb_name))
        train_pred = self.cls.predict(self.train_features)
        test_pred = self.cls.predict(self.test_features)
        print("Accuracy train-set: {}".format(metrics.accuracy_score(self.train_labels, train_pred)))
        print("Accuracy test-set: {}".format(metrics.accuracy_score(self.test_labels, test_pred)))
        y_true, y_pred = self.test_labels, self.cls.predict(self.test_features)
        print(classification_report(y_true, y_pred))
        # cross validation
        scores = cross_val_score(self.cls, self.train_features, self.train_labels, cv=10)
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        # Plot non-normalized confusion matrix

        plot_confusion_matrix(self.test_labels, y_pred, classes=self.getUniqueValue(),
                              title='Confusion matrix, without normalization')

class Img(Model):

    @property
    def name_label(self):
        a = pd.DataFrame(data=[self.name,self.label],
                         columns=["Image","Label"])
        return a

    class PipeStep(object):
        """
        Wrapper for turning functions into pipeline transforms (no-fitting)
        """

        def __init__(self, step_func):
            self._step_func = step_func

        def fit(self, *args):
            return self

        def transform(self, X):
            return self._step_func(X)


    def definition(self,*args, **kwars):
        self.cls = args[0]
        #lime
        if(len(args) == 6 and isinstance(args[5], dict)):
            self.session = args[1]  #cls keras.engine.triaing.model    tensorflow.python.session
            self.im = args[2]  #image as ndarray
            self.processed_images = args[3]  #placeholder with high,width,channel
            self.probabilities = args[4] #tensor Softmax
            self.names = args[5]        #names of labels
            self.segmenter = None
            #deepexplainer with normal jpeg image
        elif(len(args) == 7):
            self.session = args[1]
            self.xi = args[2]
            self.labels = args[3]
            self.logits = args[4]
            self.X = args[5]
            self.im = args[6]
        elif(len(args) == 5):  #deep explainer with mnist
            self.session = args[0]
            self.logits = args[1]
            self.yi = args[2]
            self.X = args[3]
            self.xi = args[4]
            #saliency
        elif(len(args) == 8):
            #session è lo stesso di cls
            self.session = args[0]
            self.processed_images = args[1]
            self.graph = args[2]
            self.neuron_selector = args[3]
            self.im = args[5]
            self.prediction_class = self.session.run(args[6], feed_dict = {self.processed_images: [self.im]})[0]
            self.y = args[7]
        else:
            print("Bad implemented")

    def predict_fn(self):
        return self.session.run(self.probabilities, feed_dict={self.processed_images: self.im})

    def set_pipiline(self):
        makegray_step = Img.PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
        flatten_step = Img.PipeStep(lambda img_list: [img.ravel() for img in img_list])
        self.simple_rf_pipeline = Pipeline([
            ('Make Gray', makegray_step),
            ('Flatten Image', flatten_step),
             ('Normalize', Normalizer()),
             ('PCA', PCA(16)),
            ('RF', RandomForestClassifier())
        ])
