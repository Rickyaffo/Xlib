from sklearn.model_selection import train_test_split
import pickle
from LOREM.code.util import record2str
import numpy as np
import logging
from pprint import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
import seaborn as sns
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Models:

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

class Tab(Models):

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

    def getClsFromFile(self,filename):
        with open(filename, 'rb') as f:
            self.cls = pickle.load(f)
        return self.cls

    def getClsParams(self):
        pprint(self.cls.get_params())

    def set_idx(self,df):
        columns_tmp = list(df.columns)
        columns_tmp.remove(self.class_names)
        self.idx_features = {i: col for i, col in enumerate(columns_tmp)}

    def values(self,target = "class"):
        return self.df.loc[:, self.df.columns != target].values

    def getValues(self,df,name):
        df.columns.get_loc(name)

    def getUniqueValue(self,name = "class"):
        return self.df[name].unique()

    def setSplit(self,X,Y,train_size=0.70):
        self.X = X
        self.Y = Y
        self.split(X,Y,train_size)

    def split(self,X,Y,train_size):
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(X, Y, train_size=train_size, random_state= self.seed, stratify = self.Y)   #necessary to avoid using different seeds and, consequently, different random splits

    def set_features_names(self):
        self.feature_names = [obj for i, obj in enumerate(self.df.columns) if i < len(self.df.columns) - 1]

    def set_class_names(self):
        self.class_names = [obj for i, obj in enumerate(self.df.columns) if i == len(self.df.columns) - 1]

    def get_cls(self):
        return self.cls

    def set_cls(self, x):
        if(isinstance(x,MLPClassifier)):
            self.bb_name = "Neural Network"
        elif(isinstance(x,SVC)):
            self.bb_name = "Support Vector Machine"
        else:
            self.bb_name ="Random Forest"
        self.cls = x

    def choose_features(self, n_features):
        return self.features_names[n_features]

    def toPredict(self,row, prediction):
        print('Prediction(x) = { %s }' % prediction)
        print('x = %s' % record2str(self.test_features[row], self.features_names, self.numeric_columns))
        print('')

    def f1_score(self):
        preds = self.cls.predict(self.test_features)
        f1 = f1_score(self.test_labels, preds)
        print("F1 for {0}: {1}".format("b", f1))

    def variable_importance(self):
        self.interpreter = Interpretation(self.test_features, feature_names=self.features_names)
        self.model_global = InMemoryModel(self.cls.predict_proba, examples=self.train_features,target_names=self.class_values)
        plots = self.interpreter.feature_importance.plot_feature_importance(self.model_global, ascending=False)

    def variable_dependencies(self,x=1):
        # Use partial dependence to understand the relationship between a variable and a model's predictions
        print("x: {} ".format(self.features_names[x]))
        axes_list = self.interpreter.partial_dependence.plot_partial_dependence([self.features_names[x]], self.model_global,
                                                                           grid_resolution=30,
                                                                           with_variance=True,
                                                                           figsize=(10, 5))
        ax = axes_list[0][1]
        try:
            ax.set_title(self.bb_name)
            ax.set_ylim(0, 1)
        except AttributeError:
            ax.suptitle(self.bb_name)

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

    #todo test set training set
    def open_dataset(self,feature = "class"):
        df = self.df
        try:
            if(feature == "class"):
                feature = self.class_name
                sns.countplot(x=feature, data=df)
            else:
                if(len(self.getUniqueValue(feature)) > 4 and len(self.getUniqueValue(feature)) < 100):
                    g = sns.FacetGrid(df, col=self.class_name, size=3, aspect=2)
                    g.map(plt.hist, feature, bins =int(len(self.getUniqueValue(feature)) / 3) , color = 'r')
                else:
                    print("The column is numerical continuous.")
        except (ValueError,KeyError):
            print("Attribute not present in the dataframe.")

    def __getattr__(self, attr):
        logger.debug("Value {} non present in the model.".format(attr))
        return None
