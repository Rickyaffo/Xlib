'''
01 - 10 - 2018
@author: Affolter Riccardo
'''
import cv2
import glob
from deepexplain.tensorflow import DeepExplain
from scipy.misc import imread
import json
from algorithms.preprocessing import *
from tensorflow.contrib.slim.python.slim.nets import inception
from datasets import imagenet
from utility.utility import *
import _pickle as cPickle
from keras.models import model_from_json
from algorithms import Explainer as ex
import os
from algorithms.BlackBox import *
from algorithms.Explainer import DeepExplainer
from utility.Constants import Constants
import gzip
from algorithms import BlackBox
from pathlib import Path
from tf_models.slim.nets import inception
from preprocessing import inception_preprocessing
from keras.applications import inception_v3 as kerasinception_v3
from tensorflow.contrib.slim.python.slim.nets import inception_v3

np.random.seed(1)

class InputFile():
    def __init__(self, filename):
        self.filename = filename

    def __getattr__(self, attr):
        logger.debug("Value {} non present in the File.".format(attr))
        return None

class TabFile(InputFile):
    def __init__(self, filename):
        PATH = Constants.PATH.replace("\\src\\ui", "\\Dataset\\") if "\\src\\ui" in Constants.PATH else Constants.PATH.replace("\\src", "\\Dataset\\")
        super().__init__(PATH + str(filename.split(".")[0] + ".csv"))
        self.name = filename.split(".")[0]
        self.dictTabModel = {}
        self.metadata = PATH + str(filename.split(".")[0] + ".names.txt")

    def play(self,blackbox = "RandomForest"):
        def getBB(model):
            if (Path(str(self.filename.split(".")[0] + '_' + model.name + '.pickle')).exists()):
                cls = model.getClsFromFile(str(self.filename.split(".")[0] + '_' + model.name + '.pickle'))
                model.set_cls(cls)
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
            return model
        print("file {} ".format(self.name))
        column = ['c_jail_in', 'c_jail_out', 'decile_score', 'score_text', 'education-num', 'fnlwgt']
        if(blackbox  not in  self.dictTabModel):
            model = preparewithNames(self.name,self.metadata,self.filename,column)
            model.filename = self.filename
            model.name = blackbox
            X,Y = model.df[model.features_names].values, model.df[model.class_name].values
            model.setSplit(X, Y)
            model = getBB(model)
            if (model.dictTabExplainer is None):
                model.dictTabExplainer = {x: getattr(ex, x)(self.filename) for x in Constants.EXPLAINERTAB}
            self.dictTabModel[blackbox] = model
  #      self.explain("Anchor",blackbox,6)



    def explain(self,ex,bb,row):
        model = self.dictTabModel[bb]
        explainer = model.dictTabExplainer[ex]
        explainer.setRowExplained(model, row)
        explainer.explain(model)
        self.dictTabModel[bb].dictTabExplainer[ex] = explainer

class ImgFile(InputFile):

    def __init__(self, filename):
        super().__init__(filename)
        self.ext = "jpg"
        self.dictImgExplainer = {x: getattr(ex, x)(self.filename) for x in Constants.EXPLAINERIMG}
        PATH = Constants.PATH.replace("\\src\\ui", "\\Dataset\\") if "\\src\\ui" in Constants.PATH else Constants.PATH.replace("\\src", "\\Dataset\\")
        self.PATH = PATH.replace("\\", "/")
        self.dictImgModel = {}

    def play(self,ex = "DeepExplainer"):
        print("file {} ".format(self.filename))
        self.explainer = self.dictImgExplainer[ex]
        if(ex not in self.dictImgModel):
            self.img = self.prepareImage(self.filename)
            self.dictImgModel[ex] = self.img
   #     self.explain(ex,"intgrad")

    def explain(self, ex, method=1):
        model = self.dictImgModel[ex]
        self.explainer.explain(model, method)
        model.explainer[method] = self.explainer

    def LoadImage(self,file_path):
        im = cv2.imread(file_path)
        im = cv2.resize(im, (299, 299), 3)
        return im / 127.5 - 1.0

    def createBB(self):
         if (Constants.BLACKBOX[3] == self.img.name):
             bb = BlackBox.CNN(self.img)
             bb.cls.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])
             bb.fit(self.img)
             img = bb.start(self.img)
             model_json = img.cls.to_json()
             with open(self.PATH + "models/model.json", "w") as json_file:
                 json_file.write(model_json)
                 # serialize weights to HDF5
                 img.cls.save_weights(self.PATH + "models/model.h5")
             print("Saved model in models")
         else:
             json_file = open(self.PATH + "models/model.json", 'r')
             loaded_model_json = json_file.read()
             json_file.close()
             loaded_model = model_from_json(loaded_model_json)
             # load weights into new model
             loaded_model.load_weights(self.PATH + "models/model.h5")
             print("Loaded model from disk")
             bb = BlackBox.CNN(self.img)
             bb.cls = loaded_model
             bb.cls.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])
             self.img.set_cls(loaded_model)
         bb.evaluate(img)

    def prepareImage(self, name):
        self.filesTest = {}
        PATH = self.PATH + ("chosen_1000_images/")
        slim = tf.contrib.slim
        session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        graph = tf.Graph()
        processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
        checkpoints_dir = PATH.replace("chosen_1000_images/", "models/")
        img = Img()
        if (isinstance(self.explainer, ex.Lime)):
            sys.path.append(PATH.replace("Dataset/chosen_1000_images/", "src/tf_models/slim/"))  #todo replace tensorflow
            def transform_img_fn(path_list):
                out = []
                for f in path_list:
                    with open(f, 'rb') as t:
                        contents = t.read()
                    image_raw = tf.image.decode_jpeg(contents, channels=3)
                    image = inception_preprocessing.preprocess_image(image_raw, 299, 299,
                                                                     is_training=False)
                out.append(image)
                return session.run([out])[0]
            names = imagenet.create_readable_names_for_imagenet_labels()
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, _ = inception.inception_v3(processed_images,reuse=tf.AUTO_REUSE, num_classes=1001,
                                                   is_training=False)
            probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
                slim.get_model_variables('InceptionV3'))
            session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            init_fn(session)
            bb = kerasinception_v3.InceptionV3()
            images = transform_img_fn([PATH + self.filename])
            img.definition(bb, session, images, processed_images, probabilities, names)
            img.x = printPrediction(img)
        elif (isinstance(self.explainer, ex.Lore)):
            imlist = list()
            i = 0;
            row = None
            for filename in sorted(glob.glob(PATH + '*.JPEG')):
                if (name == filename.split("\\")[1]):
                    row = i
                image = cv2.imread(filename)
                imlist.append(image)
                i += 1
            if (row is None):
                print("File not Found")
                return
            class_map = json.load(open(PATH + 'imagenet_class_index.json'))
            class_map = {int(k): v for k, v in class_map.items()}
            inv_class_map = {v[0]: k for k, v in class_map.items()}
            class_values = [''] * len(class_map)
            for k, v in class_map.items():
                class_values[k] = v[1]
            class_name = 'class'
            bb = kerasinception_v3.InceptionV3()
            img.definition(bb, inv_class_map, class_values, class_name, imlist[row], imlist, row)
        elif (isinstance(self.explainer, ex.DeepExplainer)):
            with DeepExplain(session=session, graph=session.graph) as de:
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    _, end_points = inception.inception_v3(processed_images, reuse=tf.AUTO_REUSE,num_classes=1001, is_training=False)
                logits = end_points['Logits']
                yi = tf.argmax(logits, 1)
                saver = tf.train.Saver(slim.get_model_variables())
                saver.restore(session, checkpoints_dir + "inception_v3.ckpt")
                filenames, xs = self.load_images(self.PATH)
                labels = session.run(yi, feed_dict={processed_images: xs})
                print(self.filename, labels)
                img.definition(end_points,session, xs, labels,logits,processed_images)
                im = self.LoadImage(PATH + self.filename)
                img.im = im
        elif (isinstance(self.explainer, ex.Saliency)):
            im = self.LoadImage(PATH + self.filename)
            with graph.as_default():
                processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

                with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                    _, end_points = inception_v3.inception_v3(processed_images, is_training=False, num_classes=1001)

                    # Restore the checkpoint
                    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True),graph=graph)
                    saver = tf.train.Saver()
                    saver.restore(sess, checkpoints_dir + "inception_v3.ckpt")

                # Construct the scalar neuron tensor.
                logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
                neuron_selector = tf.placeholder(tf.int32)
                y = logits[0][neuron_selector]

                # Construct tensor for predictions.
                prediction = tf.argmax(logits, 1)
                # Load the image
            img.definition(sess, processed_images, graph, neuron_selector, logits, im, prediction, y)
        return img

    def load_images(self,PATH):
        images = np.zeros((1, 299, 299, 3))
        filenames = []
        idx = 0
        for filepath in tf.gfile.Glob(os.path.join(PATH, '*.JPEG')):
            if(filepath == self.filename):
                with tf.gfile.Open(filepath, 'rb') as f:
                    image = imread(f, mode='RGB').astype(np.float) / 255.0
                # Images for inception classifier are normalized to be in [-1, 1] interval.
                images[idx, :, :, :] = image * 2.0 - 1.0
                filenames.append(os.path.basename(filepath))
                idx += 1
        return filenames, images


class TextFile(InputFile):
    ext = ".txt"
    def __init__(self, filename):
        super().__init__(filename)
        self.ext = "txt"
    def play(self):
        pass ##TODO

