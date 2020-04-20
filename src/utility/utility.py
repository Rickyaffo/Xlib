#from algorithms.Explainer import *
import pandas as pd

import random
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as P
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uncertainties import ufloat
from utility.Constants import Constants
import seaborn as sns
import glob

def printDF(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def mean(label,arr, c=None):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    mn_std = ufloat(mean, std)
    print("Mean {} : {}".format(label, mn_std))

def name_image():
    PATH = os.path.abspath(os.curdir)
    filesTest = {}
    path = PATH.replace("\\src", "\\Dataset\\")
    path = path.replace("\\", "/")
    path = path + "/images/"
    for fname in glob.glob(os.path.join(path, '*.jpeg')):
        fname = fname.replace("\\", "/")
        subpath = fname.split("/")[-1]
        filesTest[fname] = subpath
    df = pd.DataFrame({'name': list(filesTest.values())})
    return df

def calculateCoherence(arrRules,a,d):
    jaccard = {v: [] for k, v in d.items()}
    rushl = {v: [] for k, v in d.items()}
    rules = []
    [rules.append([]) for x in range(2)]
    intersect = lambda x, y: len(list(x.intersection(y)))
    union = lambda x, y: len(list(x.union(y)))
    coh = lambda x, arr, r, coherence: [arr[v].append(coherence) for k, v in x.items() if v == str(r)]
    rushlength = lambda x, arr, r, rl: [arr[v].append(rl) for k, v in x.items() if v == str(r)]
    if (a.label == "Lore"):
        for rul in range(len(arrRules[0])):
            if(len(arrRules[0]) > 0):
                if(isinstance(arrRules[0][rul],str)):
                    rules[0].append(["No rule"])
                    rules[1].append(arrRules[1][rul])
                else:
                    str_rule = arrRules[0][rul]
                    r = str(str_rule).replace("}", "").replace("{", "").replace("-->", "").replace(
                        "class: " + str(str_rule.cons), "").rstrip().lstrip().split(", ")
                    rules[0].append(r)
                    rules[1].append(arrRules[1][rul])
        len_rules = len(rules[0])
    elif (a.label == "Anchor" or a.label == "Lime"):
        for rul in range(len(arrRules[0])):
            if (len(arrRules[0][rul]) > 0):
                str_rule = arrRules[0][rul]
                r = str(str_rule).replace("  }", "").replace("{ ", "").rstrip().lstrip().split(" ---> ")
                rules[0].append(r[0].split(" , "))
                rules[1].append(arrRules[1][rul])
        len_rules = len(rules[0])
    for rul in range(len_rules):
        coherence = [
            intersect(set(rules[0][rul]), set(rules[0][i])) / union(set(rules[0][rul]), set(rules[0][i]))
            for i in range(len_rules) if rules[1][rul] == rules[1][i] and rules[0][rul] != "No rule" and rul != i]
        coherence = np.mean(coherence) if len(coherence) > 0 else 0
        rl = [min(len(rules[0][rul]), len(rules[0][i])) / max(len(rules[0][rul]), len(rules[0][i]))
              for i in range(len_rules) if rules[1][rul] == rules[1][i] and rules[0][rul] != "No rule" and rul != i]
        rl = np.mean(rl) if len(rl) > 0 else 0
        coh(d, jaccard, rules[1][rul], coherence)
        rushlength(d, rushl, rules[1][rul], rl)
    return jaccard,rushl

def getArgs(method,display = False,num_features = 5):
    if (method is None):
        methodD = "grad*input"
        methodS = "Vanilla Gradient"
    else:
        if (method in ["Vanilla Gradient", "Guided Backprop", 'Integrated Gradients', 'Occlusion']):
            methodS = method
            methodD = "grad*input"
        else:
            methodD = method;
            methodS = "Vanilla Gradient";
    return {"display" : display, "num_features": num_features, "methodD": methodD,
              "methodS": methodS}

def boxPlotExplanation(metric,*args,second_class = False):
    bplt = {}
    j = 0
    for listP in args:
        i = 0
        arr = []
        arrRulesForCoherence = []
        [arrRulesForCoherence.append([]) for x in range(2)]
        if(not isinstance(listP,list)):
            listP = [listP for i in range(1)]
        for a in listP:
            print
            label = a.label
     #       color = a.color
            if(metric == "Time"):
                arr.append(a.Time)
            elif(metric == "Length"):
                arr.append(a.Length)
            elif(metric == "Fidelity"):
                arr.append(a.Fidelity)
            elif(metric == "Coherence"):
                arrRulesForCoherence[0].append(a.Rules)
                arrRulesForCoherence[1].append(a.target)
                if(i == len(listP)-1):
                    d = {k: str(v) for k, v in a.class_values.items()}
                    jac, rus = calculateCoherence(arrRulesForCoherence, a, d)
                    label1 = "J "
                    label2 = "CoefD "
                    if (second_class == True):
                        for k, v in d.items():
                            bplt[label1 + v] = jac[v]
                            mean(label1, jac[v])
                            bplt[label2 + v] = rus[v]
                            mean(label2, rus[v])
                    else:
                        bplt[label + " " + label1] = jac[d[0]]
                        mean(label + " " + label1, jac[d[0]])
                        bplt[label + " " +  label2] = rus[d[0]]
                        mean(label + " " +  label2, rus[d[0]])
                i += 1
        if (metric != "Coherence"):
            bplt[label + str(j)] = arr
            mean(label, arr)
        j += 1
    showStat(bplt,metric)

def getLocallyInstance(n,dataset,PATH):
    idx = 0
    filenames = []
    if(dataset == None):
        for filepath in tf.gfile.Glob(os.path.join(PATH, '*.JPEG')):
            filenames.append(os.path.basename(filepath))
            idx += 1
            if(idx == n):
                 break
    else:
        for i in range(n):
            filenames.append(i)
    return filenames

def showStat(bplt,metric):
    if (metric != "Coherence"):
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.boxplot(bplt.values())
    ax.set_xticklabels(bplt.keys())
 #   plt.xlabel(metric)
    plt.ylabel(metric)
    plt.show()

def open_dataset(model,feature = "class"):
    df = model.df
    try:
        if(feature == "class"):
            feature = model.class_name
            sns.countplot(x=feature, data=df)
        else:
            if(len(model.getUniqueValue(feature)) == 2):
                g = sns.FacetGrid(df, col=model.class_name, size=3, aspect=2)
                g.map(plt.hist, feature, bins =1, color = 'r')
            elif(len(model.getUniqueValue(feature)) < 100):
                g = sns.FacetGrid(df, col=model.class_name, size=3, aspect=2)
                g.map(plt.hist, feature, bins=int(len(model.getUniqueValue(feature)) / 3),  color='r')
            else:
                print("The column is numerical continuous.")
    except (ValueError,KeyError) as ex:
        print("Attribute not present in the dataframe : {}".format(ex))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

  #v  print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    im = ((im + 1) * 127.5).astype(np.uint8)
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayScaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def printPrediction(model, display = True):
#    plt.imshow(model.im[0] / 2 + 0.5)
    preds = model.predict_fn()
    for x in preds.argsort()[0][-5:]:
        if (display):
            print(x, model.names[x], preds[0, x])
    return x

def ExplainerTab(fun,dictTabExplainer):
    if (fun() == Constants.EXPLAINERTAB[0]):
        a = dictTabExplainer[Constants.EXPLAINERTAB[0]]
    elif (fun() == Constants.EXPLAINERTAB[1]):
        a = dictTabExplainer[Constants.EXPLAINERTAB[1]]
    elif(fun() == Constants.EXPLAINERTAB[2]):
        a = dictTabExplainer[Constants.EXPLAINERTAB[2]]
    return a

def ExplainerImg(fun,dictImgExplainer):
    if (fun() == Constants.EXPLAINERIMG[0]):
        a = dictImgExplainer[Constants.EXPLAINERIMG[0]]
    elif (fun() == Constants.EXPLAINERIMG[1]):
        a = dictImgExplainer[Constants.EXPLAINERIMG[1]]
    elif (fun() == Constants.EXPLAINERIMG[2]):
        a = dictImgExplainer[Constants.EXPLAINERIMG[2]]
    elif (fun() == Constants.EXPLAINERIMG[3]):
        a = dictImgExplainer[Constants.EXPLAINERIMG[3]]
    return a

def getTestSet(model):
    test_set = []
    test_set.append([])
    test_set.append([])
    n = 250
    ran = random.sample(range(model.test_labels.shape[0]), model.test_labels.shape[0])
    if (model.test_labels.shape[0] > 250):
        n = 110
    tot = n
    for i in ran:
        if(tot > n/7):
            test_set[0].append(i)
            test_set[1].append(model.test_labels[i])
            tot = tot - 1
        else:
            a = np.array(test_set[1])
            unique, counts = np.unique(a, return_counts=True)
            bil = dict(zip(unique, counts))
            value = max(bil, key=lambda x: bil.get(x))
            if(value != model.test_labels[i]):
                test_set[0].append(i)
                test_set[1].append(model.test_labels[i])
                tot = tot - 1
        if(tot == 0 ):
            break
  #  print(bil)
    return test_set

def ShowDivergingImage(grad, title='', percentile=99, ax=None):
    if ax is None:
        fig, ax = P.subplots()
    else:
        fig = ax.figure

    P.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err



