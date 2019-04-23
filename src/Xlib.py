'''
01 - 10 - 2018
@author: Affolter Riccardo
'''
import sys,os
sys.path.append(os.path.dirname(__file__) + '\\tf_models\\slim\\')
import tensorflow as tf
from algorithms.KDD import preparewithNames,prepareImage
from utility.Utility import getTestSet, boxPlotExplanation, open_dataset, name_image, getLocallyInstance, getArgs
from algorithms import Explainer as ex
from algorithms.BlackBox import *
from utility.Constants import Constants
from skimage.color import gray2rgb
from pathlib import Path
import progressbar
from time import sleep
np.random.seed(1)

def getBB(model, path_file):
    """set the classifier to the model if it already exists,
     otherwise compute a new one."""
    if (Path(path_file).exists()):
        model.set_cls(model.getModelFromFile(path_file))
    else:
        if (Constants.BLACKBOX[0] == model.name):
            m = RandomForest(model.seed)
            model = m.start(model)
        if (Constants.BLACKBOX[1] == model.name):
            m = NeuralN(model.seed)
            model = m.start(model)
        if (Constants.BLACKBOX[2] == model.name):
            m = Svm(model.seed)
            model = m.start(model)
        if (Constants.BLACKBOX[3] == model.name):
            m = CNN(model.seed)
            model = m.start(model)
    return model


class InputFile():
    def __init__(self, dataset = None):
        self.dataset = dataset

    def __getattr__(self, attr):
        return None

class TabFile(InputFile):
    def __init__(self, filename):
        PATH = Constants.PATH.replace("\\src", "\\Dataset\\")
        super().__init__(PATH + str(filename.split(".")[0] + ".csv"))
        self.name = filename.split(".")[0]
        self.dictTabModel = {}
        self.metadata = PATH + str(filename.split(".")[0] + ".names.txt")

    def display(self,ex,bb):
        explainer = self.dictTabModel[bb].dictTabLocalExplainer[ex]
        explainer.display(self.dictTabModel[bb])

    def evaluate(self, bb):
        self.dictTabModel[bb].evaluate()
        self.dictTabModel[bb].f1_score()

    @property
    def SVM(self):
        return self.dictTabModel["SVM"]

    @property
    def RF(self):
        return self.dictTabModel["RandomForest"]

    @property
    def NN(self):
        return self.dictTabModel["Neural_Network"]

    def play(self,blackbox = "RandomForest"):
        def model(name, metadata, dataset):
            column = ['c_jail_in', 'c_jail_out', 'decile_score', 'score_text', 'education-num', 'fnlwgt']
            datamodel = preparewithNames(name, metadata, dataset, column)
            datamodel.filename = dataset
            X, Y = datamodel.df[datamodel.features_names].values, datamodel.df[datamodel.class_name].values
            datamodel.setSplit(X, Y)
            return datamodel
        if(blackbox  not in self.dictTabModel):
            datamodel = model(self.name, self.metadata, self.dataset)
            datamodel.name = blackbox
            datamodel = getBB(datamodel,str(self.dataset.split(".")[0] + '_' + datamodel.name + '.pickle'))
            if(self.test_set is None):
                self.test_set = getTestSet(datamodel)
            datamodel.dictTabLocalExplainer = {x: getattr(ex, x)(self.dataset) for x in Constants.EXPLAINERTABLOCAL}
            datamodel.dictTabGlobalExplainer = {x: getattr(ex, x)(self.dataset) for x in Constants.EXPLAINERTABGLOBAL}
            self.dictTabModel[blackbox] = datamodel

    def explainLocal(self, ex, bb, row, existing=True, display=False, num_features=5):
        kwargs = getArgs(None,display=display,num_features=num_features)
        try:
            model = self.dictTabModel[bb]
            explainer = model.dictTabLocalExplainer[ex]

            if(explainer.setRowExplained(model, row,existing)):
                explainer.explain(model, **kwargs)
                self.dictTabModel[bb].dictTabLocalExplainer[ex] = explainer
        except KeyError as error:
            # Output expected KeyErrors.
            print("Explainer: {} or Black box: {} wrong".format(bb,ex))
        return explainer.properties

    def explainGlobal(self,ex,bb,variables=-1):
        model = self.dictTabModel[bb]
        explainer = model.dictTabGlobalExplainer[ex]
        explainer.explain(model,variables)

    def explainTestLocal(self,bb,ex ,display=False, num_features = 5):
        print(len(self.test_set[0]))
        test_set = self.test_set
        model = self.dictTabModel[bb]
        experiment_result = []
        kwargs = {"display": display, "num_features" : num_features, "methodD": "grad*input", "methodS": "Vanilla Gradient"}
        explainer = model.dictTabLocalExplainer[ex]
        bar = progressbar.ProgressBar(maxval=len(self.test_set[0]) + 10, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        i = 0
        for row in test_set[0]:
            explainer.setRowExplained(model, row, True)
            experiment_result.append(explainer.explain(model, **kwargs))
            i += 1
            bar.update(i + 1)
            sleep(0.1)
        bar.finish()
        return experiment_result

class ImgFile(InputFile):

    def __init__(self, filename = None, dataset = None):
        super().__init__(filename)
        self.__initialization(filename,dataset)

    def display(self,ex,bb= "CNN"):
        explainer = self.dictImgDisplay[ex]
        explainer.display(self.dictImgModel[self.ex])

    @property
    def CNN(self):
        return [x for x in self.dictImgModel][0]

    def __initialization(self,filename,dataset = None):
        self.name = filename
        self.dataset = dataset
        self.dictImgModel = {}
        self.dictImgDisplay = {}
        PATH = Constants.PATH.replace("\\src\\ui",
                                      "\\Dataset\\") if "\\src\\ui" in Constants.PATH else Constants.PATH.replace(
            "\\src", "\\Dataset\\")
        self.PATH = PATH.replace("\\", "/")
        PATH = self.PATH + ("images/")
        if (self.dataset is not None):
            if (self.dataset == "mnist"):
                self.SHAPE_1, self.SHAPE_2 = 28, 28
                from sklearn.datasets import fetch_mldata
                dataset = fetch_mldata('MNIST original')
                self.X_vec = np.stack(
                    [gray2rgb(iimg) for iimg in dataset.data.reshape((-1, self.SHAPE_1, self.SHAPE_2))], 0)
                self.y_vec = dataset.target.astype(np.uint8)
            else:
                print("Name of the dataset wrong or not already available.")
        else:
            self.class_values = {}
            with open(os.path.join(PATH, 'ILSVRC2012_validation_ground_truth.txt')) as f:
                groundtruths = f.readlines()
            groundtruths = [int(x.strip()) for x in groundtruths]
            for filepath in tf.gfile.Glob(os.path.join(PATH, '*.JPEG')):
                num_image = ''.join(i for i in os.path.basename(filepath) if i.isdigit())[4:]
                self.class_values[os.path.basename(filepath)] = groundtruths[int(num_image)]
            self.SHAPE_1, self.SHAPE_2 = 299, 299
        self.dictImgExplainer = {x: getattr(ex, x)(self.dataset) for x in
                                 Constants.EXPLAINERIMG}

    def getLabel(self):
        if (isinstance(self.class_values, dict)):
            label = self.class_values[self.name]
        else:
            label = None
        return label

    def play(self,ex):
        self.ex = ex
        label = self.getLabel()
        if (ex not in self.dictImgModel):
            explainer = self.dictImgExplainer[ex]
            img = prepareImage(self.name,self.PATH,self.SHAPE_1,self.SHAPE_2,self.dataset,self.X_vec,self.y_vec,explainer,label)
            self.dictImgModel[ex] = img

    def explainLocal(self, method = None, display = False):
        try:
            img = self.dictImgModel[self.ex]
            kwargs = getArgs(method,display)
            explainer = self.dictImgExplainer[self.ex]
            explainer.explain(img,**kwargs)
            self.dictImgModel[self.ex] = img
            if(method is None):
                self.dictImgDisplay[self.ex] = explainer
            else:
                self.dictImgDisplay[method] = explainer
        except KeyError as error:
            # Output expected KeyErrors.
            print("Explainer: {}  wrong".format(self.ex))
        return explainer.properties

    def explainTestLocal(self,ex,method=None, n=30):
        filenames = getLocallyInstance(n,self.dataset, self.PATH + ("images/"))
        experiment_result = []
        explainer = self.dictImgExplainer[ex]
        kwargs = getArgs(method,False)
        for i in range(n):
            self.name = filenames[i]
            label = self.getLabel()
            self.__initialization(filenames[i])
            img = prepareImage(filenames[i],self.PATH,self.SHAPE_1,self.SHAPE_2,self.dataset,self.X_vec,self.y_vec,explainer,label)
            experiment_result.append(explainer.explain(img,**kwargs))
        return experiment_result

class TextFile(InputFile):
    def __init__(self, filename):
        super().__init__(filename)
    def play(self):
        pass ##TODO

