'''
01 - 10 - 2018
@author: Affolter Riccardo
'''
from datamodel.DataModel import Tab, Img
from algorithms.Properties import Properties
from utility.Constants import Constants
import cv2,time,warnings
import numpy as np

import lime.lime_tabular
from deepexplain.tensorflow import DeepExplain
from LOREM.code.realise import REALISE
from LOREM.code.lorem import LOREM
from LOREM.code.util import neuclidean, multilabel2str
from anchor import anchor_tabular
import saliency
from lime import lime_image
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

import tensorflow as tf
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

class Explainer:

    def __init__(self,filename = None):
        self.filename = filename

    def __getattr__(self, attr):
            return None

    def addImageRules(self):
        self.rulesImages = []
        self.rulesImages.append([])
        self.rulesImages[0].append([])
        self.rulesImages[0].append([])
        self.rulesImages[0].append([])

    def setRowExplained(self,datamodel,row,existing):
        if not existing:
            if(len(datamodel.features_names) == len(row)):
                self.row = np.array(row) ; return True
            else:
                print("Dimension of the row wrong.")
                return False
        else:
            self.row = datamodel.test_features[row] ; return True

    def setImgExplained(self,img):
        self.addImageRules()
        self.rulesImages[0][0].append(img / 2 + 0.5)
        self.rulesImages[0][1].append("")
        self.rulesImages[0][2].append("Original")

    def displayTab(self):
        self.properties.displayTab()

    def displayImg(self):
        self.properties.displayImg()

    def explain(self, datamodel,num_features,display, methodD , methodS):
        if(isinstance(datamodel,Tab)):
            return self.explainTab(datamodel,display)
        elif(isinstance(datamodel,Img)):
            return self.explainImg(datamodel,display,methodD,methodS)

    def newProperties(self):
        self.properties = Properties()

    def insertPropertiesTab(self,rules, num_features, time, c, label, fidelity, target, class_values):
        self.properties.Rules = rules
        self.properties.Length = num_features
        self.properties.Time = time
        self.properties.color = c
        self.properties.label = label
        self.properties.Fidelity = fidelity
        self.properties.target = target
        self.properties.class_values = class_values

    def insertPropertiesImg(self,img2show,mask,title, time, c, label,exp, shape1, shape2):
        self.properties.Time = time
        self.properties.color = c
        self.properties.label = label
        self.properties.exp = exp
        self.properties.SHAPE1 = shape1
        self.properties.SHAPE2 = shape2
        self.rulesImages[0][0].append(img2show)
        self.rulesImages[0][1].append(mask)
        self.rulesImages[0][2].append(title)
        self.properties.rulesImages = self.rulesImages

class Skater(Explainer):

    def explain(self, model,variables = None ):
        self.variable_importance(model,variables) if variables != None  else self.variable_importance(model)

    def variable_importance(self,model,variables):
        self.interpreter = Interpretation(model.test_features, feature_names=model.features_names)
        self.model_global = InMemoryModel(model.cls.predict_proba, examples=model.train_features ,target_names=model.class_values)
        f, ax = self.interpreter.feature_importance.plot_feature_importance(self.model_global, ascending=False)
        if(len(model.df.columns) > 15):
            ax.set_ylim([0, 15])
        ax.set_xlabel("\u03B5")
        if variables != -1:
            self.variable_dependencies(model,variables)

    def variable_dependencies(self,model,x):
        axes_list = self.interpreter.partial_dependence.plot_partial_dependence([model.features_names[x]], self.model_global, n_samples=100,
                                                                           grid_resolution=30,
                                                                           with_variance=True,
                                                                           figsize=(10, 5))
        ax = axes_list[0][1]
        try:
            ax.set_title(model.bb_name)
            ax.set_ylim(0, 1)
        except AttributeError:
            ax.suptitle(model.bb_name)
        ax.set_ylabel("Impact ({})".format(model.features_names[x]))

class Lime(Explainer):

    def explain(self,model,num_features,display,methodD,methodS):
        if (isinstance(model, Tab)):
            return self.explainTab(model, display,num_features)
        elif (isinstance(model, Img)):
            return self.explainImg(model, display)

    def display(self, model):
        if (isinstance(model, Tab)):
            self.explanation.show_in_notebook(show_table=True, show_all=False)
            model.toPredict(self.row, self.properties.target)
            super().displayTab()
        elif (isinstance(model, Img)):
            super().displayImg()

    def calculateFidelity(self,bb_outcome):
        bb_probs = self.explainer.Zl[:, bb_outcome]
        lr_probs = self.explainer.lr.predict(self.explainer.Zlr)
        return 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)

    def explainTab(self,model,display,num_features):
        self.newProperties()
        cls = model.get_cls()
        if (model.name == Constants.BLACKBOX[2]):
            cls.probability = True
        predict_fn = cls.predict_proba
        t0 = time.clock()
        self.explainer = lime.lime_tabular.LimeTabularExplainer(model.train_features,
                                                                class_names=sorted(model.getUniqueValue().tolist()),
                                                                feature_names=model.features_names,
                                                                random_state=model.seed,
                                                                kernel_width=3)
        self.explanation = self.explainer.explain_instance(self.row, predict_fn, num_features=num_features)
        t = time.clock() - t0
        bb_outcome = cls.predict(self.row.reshape(1, -1))[0]
        bb_outcome_str = model.class_values[bb_outcome]
        #       label = bb_predict(np.array([X_test[i2e]]))[0]
        fidelity = self.calculateFidelity(bb_outcome)
        s = "{ "
        for n in self.explanation.as_list():
            s += n[0]
            s += " , "
        s = s[:-2] + " }"
        s +=  " ---> { " + str(bb_outcome_str) + " } "
        self.insertPropertiesTab(rules = s, num_features = num_features, time = t, c ='r', label ="Lime",
                                 fidelity = fidelity, target = str(bb_outcome_str), class_values = dict(enumerate(model.class_values)))
        if(display):
            self.display(model)
        return self.properties

    def explainImg(self,img,display):
        self.newProperties()
        if (self.filename == "mnist" or self.filename == "faces" ):
            image = img.im
            self.explainer = lime_image.LimeImageExplainer()
            t0 = time.clock()
            self.explanation = self.explainer.explain_instance(image, img.predict_fn, top_labels=10, hide_color=0,
                                                               num_samples=10000, segmentation_fn=img.segmenter)
            t = time.clock() - t0
            print("Elapsed time (s): ", time.clock() - t0)
            img2show, mask = self.explanation.get_image_and_mask(img.x, positive_only=True, num_features=5,
                                                                 hide_rest=True)
            self.setImgExplained(image)
            title = 'Pred: {}'.format(img.x)
        else:
            images = img.im; session = img.session; probabilities = img.probabilities; processed_images = img.processed_images
            def predict_fn(images):
                return session.run(probabilities, feed_dict={processed_images: images})
            self.explainer = lime_image.LimeImageExplainer()
            t0 = time.clock()
            self.explanation = self.explainer.explain_instance(images[0], predict_fn,  top_labels=10, hide_color=0,
                                                               num_samples=10000,segmentation_fn=img.segmenter)
            t = time.clock() - t0
            print("Elapsed time (s): ", time.clock() - t0)
            img2show, mask = self.explanation.get_image_and_mask(img.x, num_features=5,
                                                                 hide_rest=True)
            self.setImgExplained(img.im[0])
            title = img.names[img.x]
        self.insertPropertiesImg(img2show= img2show, mask=mask ,title=title, time=t, c='r', exp="Lime" , label="Lime",
                                 shape1=img.SHAPE1, shape2=img.SHAPE2)
        if (display):
            self.display(img)
        return self.properties

class Anchor(Explainer):

    def explainTab(self, model,display):
        self.newProperties()
        cls = model.get_cls()
        categorical = {}
        t0 = time.clock()
        self.explainer = anchor_tabular.AnchorTabularExplainer(sorted(model.getUniqueValue().tolist()), model.features_names, model.X, categorical)
        self.explainer.fit(model.train_features, model.train_labels, model.test_features, model.test_labels)
        predict_fn = lambda x: cls.predict(x)
        bb_outcome = predict_fn(self.row.reshape(1, -1))[0]
        self.explanation = self.explainer.explain_instance(self.row, cls.predict, threshold=0.95)
        t = time.clock() - t0
        rullist = self.explanation.exp_map["names"]
        if(len(rullist) == 0):
            s = "No rule"
            fidelity = 0
        else:
            s = "{ "
            for n in rullist:
                s += n
                s += " , "
            s = s[:-2] + " }"
            s += " ---> { " + str(model.class_values[bb_outcome]) + " } "
            fidelity = self.explanation.precision()
        self.insertPropertiesTab(rules=s, num_features=len(rullist), time=t, c='b', label='Anchor', fidelity=fidelity,
                                 target=str(model.class_values[bb_outcome]), class_values=dict(enumerate(model.class_values)))
        if (display):
            self.display(model)
        return self.properties

    def display(self,model):
        self.explanation.show_in_notebook()
        model.toPredict(self.row, self.properties.target)
        super().displayTab()

class Lore(Explainer):

    def explainTab(self, model,display):
        self.newProperties()
        blackbox = model.get_cls()
        def bb_predict(X):
            return blackbox.predict(X)
        bb_outcome = bb_predict(self.row.reshape(1, -1))[0]
        bb_outcome_str = model.class_values[bb_outcome] if isinstance(model.class_name, str) else multilabel2str(bb_outcome,
                                                                                                     model.class_values)
        stratify = None if isinstance(model.class_name, list) else model.rdf[model.class_name].values

        _, K, _, _ = train_test_split(model.rdf[model.real_feature_names].values, model.rdf[model.class_name].values, test_size=0.30,
                                      random_state=model.seed, stratify=stratify)
        t0 = time.clock()
        explainer = LOREM(K, bb_predict, model.features_names, model.class_name, model.class_values, model.numeric_columns, model.e,
                          neigh_type='random', categorical_use_prob=True,
                          continuous_fun_estimation=True, size=1000, ocr=0.1, multi_label=False, one_vs_rest=False,
                          random_state=model.seed, verbose=False, ngen=5)
        try:
            self.explanation = explainer.explain_instance(self.row, num_samples=1000, use_weights=True, metric=neuclidean)
            s = self.explanation.rstr()
            self.properties.CounterFactuals = self.explanation.cstr()
            length = len(self.explanation.rule.premises)
            fidelity = self.explanation.fidelity
            self.properties.CRules = self.explanation.crules
        except ValueError:
            s = "No rule"
            self.properties.CounterFactuals = ""
            length = 0
            fidelity = 0
            self.properties.CRules = []
        t = time.clock() - t0
        self.insertPropertiesTab(rules=s, num_features=length, time=t, c='g', label='Lore', fidelity=fidelity,
                                 target=str(model.class_values[bb_outcome]),
                                 class_values=dict(enumerate(model.class_values)))
        if (display):
            self.display(model)
        return self.properties

    def display(self, model):
        print('e = {\n\tr = %s\n\tc = %s    \n}' % (self.properties.Rules, self.properties.CounterFactuals))
        model.toPredict(self.row, self.properties.target)
        super().displayTab()

class DeepExplainer(Explainer):

    def explainImg(self,img,display,methodD,methodS):
        self.newProperties()
        t0 = time.clock()
        c = False
        if (self.filename == "mnist" or self.filename == "faces" ):
            self.properties.Xi = img.xi.reshape(img.im.shape[0], img.im.shape[1])
            c = True
        with DeepExplain(session=img.session) as de:
            if (methodD == "grad*input"):
                attributions = {'Gradient * Input': de.explain('grad*input',img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            elif (methodD == 'saliency'):
                attributions = { 'Saliency maps': de.explain('saliency', img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            elif (methodD == 'intgrad'):
                attributions = { 'Integrated Gradients': de.explain('intgrad', img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            elif (methodD == 'elrp'):
                attributions = { 'Epsilon-LRP': de.explain('elrp', img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            elif (methodD == 'deeplift'):
                attributions = { 'DeepLIFT (Rescale)': de.explain('deeplift', img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            elif (methodD == 'occlusion'):
                attributions = { '_Occlusion [1x1]': de.explain('occlusion', img.logits * img.yi if c else tf.reduce_max(img.logits, 1), img.X, img.xi)}
            else:
                print("Explainer wrong.")
                return
            print('Done')
        t = time.clock() - t0
        self.setImgExplained(img.im)
        for a,b in enumerate(attributions):
            explanation = attributions[b][0]
            title = b
        self.insertPropertiesImg(img2show=explanation, mask="", title=title, time=t, c='g', exp='DeepExplainer',
                                 label=methodD,
                                 shape1=img.SHAPE1, shape2=img.SHAPE2)
        if (display):
            self.display(img)
        return self.properties

    def display(self,img):
        super().displayImg()

class Saliency(Explainer):

    def explain(self,img,**kwargs):
        print("Prediction class: " + str(img.prediction_class))
        return super().explain(img,**kwargs)

    def display(self,img):
        super().displayImg()

    def explainImg(self,img,display,methodD,methodS):
        self.newProperties()
        t0 = time.clock()
        if (methodS == "Vanilla Gradient"):
            gradient_saliency = saliency.GradientSaliency(img.graph, img.session, img.y, img.processed_images)
            vanilla_mask_3d = gradient_saliency.GetMask(img.im, feed_dict={img.neuron_selector: img.prediction_class})
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img.im,
                                                                   feed_dict={img.neuron_selector: img.prediction_class})
            explanation = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
            title = 'Vanilla Gradient'
            explanation_smoothed = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
            title_smoothed = 'SmoothGrad'
        elif (methodS == 'Guided Backprop'):
            guided_backprop = saliency.GuidedBackprop(img.graph, img.session, img.y, img.processed_images)
            vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class})
            smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class})
            explanation = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop_mask_3d)
            title = 'Vanilla Guided Backprop'
            explanation_smoothed = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
            title_smoothed = 'SmoothGrad Guided Backprop'
        elif (methodS == 'Integrated Gradients'):
            baseline = saliency.np.zeros(img.im.shape)
            baseline.fill(-1)
            integrated_gradients = saliency.IntegratedGradients(img.graph, img.session, img.y, img.processed_images)
            vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class}, x_steps=25, x_baseline=baseline)
            smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class}, x_steps=25, x_baseline=baseline)
            explanation = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
            title = 'Vanilla Integrated Gradients'
            explanation_smoothed = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
            title_smoothed = 'Smoothgrad Integrated Gradients'
        elif (methodS == 'Occlusion'):
            occlusion_gradients = saliency.Occlusion(img.graph, img.session, img.y, img.processed_images)
            vanilla_occlusion_gradients_mask_3d = occlusion_gradients.GetMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class})
            smoothgrad_occlusion_gradients_mask_3d = occlusion_gradients.GetSmoothedMask(
                img.im, feed_dict={img.neuron_selector: img.prediction_class})
            explanation = saliency.VisualizeImageGrayscale(vanilla_occlusion_gradients_mask_3d)
            title = 'Vanilla Occlusion'
            explanation_smoothed = saliency.VisualizeImageGrayscale(smoothgrad_occlusion_gradients_mask_3d)
            title_smoothed = 'Smoothgrad Occlusion'
        else:
            print("Explainer wrong.")
            return
        t = time.clock() - t0
        self.setImgExplained(img.im)
        self.insertPropertiesImg(img2show=explanation, mask="", title=title, time=t, c='g', exp='Saliency',
                                 label=methodS,shape1=img.SHAPE1, shape2=img.SHAPE2)
        self.rulesImages[0][0].append(explanation_smoothed)
        self.rulesImages[0][1].append("")
        self.rulesImages[0][2].append(title_smoothed)
        if (display):
            self.display(img)
        return self.properties

class Ext(Explainer):
    def explain(self,model,**kwargs):
        print("Your own Explainer")
        # return super().explain(img,**kwargs)

    def displayImg(self):
        pass

    def displayTab(self):
        pass