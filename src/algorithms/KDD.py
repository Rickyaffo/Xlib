from algorithms import Explainer as ex
from algorithms import BlackBox
import tempfile, sys, os,glob
from collections import defaultdict
from sklearn_pandas import CategoricalImputer
#from tf_models.slim.nets import inception
from preprocessing import inception_preprocessing
from keras.applications import inception_v3 as kerasinception_v3
from datasets import imagenet
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import cv2
import tensorflow as tf
import numpy as np
from utility import Constants
from utility.Utility import printPrediction
from datamodel.DataModel import Tab,Img,train_test_split
from deepexplain.tensorflow import DeepExplain
from lime.wrappers.scikit_image import SegmentationAlgorithm
from scipy.misc import imread
import json
from keras import backend as K
from nets import inception_v3
from tensorflow.contrib.slim.python.slim.nets import inception

def get_features(metadata):
    """
    Method to give all the charateristic included in a file names.txt
     Returns
    -------
    feature_names
        a list that are the header columns
     features
     a list with name, type, continuous/discrete without class
     usecols  number of columns
    """
    features = list()
    feature_names = list()
    usecols = list()
    try:
        data = open(metadata, 'r')
        col_id = 0
        for row in data:
            field = row.strip().split(',')
            feature_names.append(field[0])
            if field[2] != 'ignore':
                usecols.append(col_id)
                if field[2] != 'class':
                    features.append((field[0], field[1], field[2]))
            col_id += 1
    except FileNotFoundError:
        print("File names not present in the Dataset directory.")
    return feature_names, features, usecols


def NanValues(df, target,feature_names, features,*args, strategy="mean"):
    if (strategy in ['mean', 'median']):
        imp = CategoricalImputer(missing_values='a_missing')
        id = ["id", "customerid"]
        fToEliminate = ["phone", "telephone"]
        fToEliminate.extend(args)
        # remove columns with constant values
        for column in df.columns:
            if df[column].unique().shape[0] == 1:
                df = df.drop(column, 1)
        for col in df.columns:
            if (col.lower() in id):
                df.set_index(col, inplace=True)
                feature_names.remove(col)
            elif (col.lower() in fToEliminate):
                df.drop(col, inplace=True, axis=1)
                feature_names.remove(col)
            else:
                if (df[col].isnull().sum() > 0):
                    if '?' in df[col].unique():
                        df[col][df[col] == '?'] = df[col].value_counts().index[0]
                    # cancello le righe se ho valori nan nella classe
                    if (col == target):
                        df.dropna(subset=[col], inplace=True)
                    else:
                        if (df[col].dtypes == np.number):
                            if (strategy == 'mean'):
                                df[col].fillna((df[col].mean()), inplace=True)
                            elif (strategy == 'median'):
                                df[col].fillna((df[col].median()), inplace=True)
                        else:
                            # categorical
                            try:
                                X = pd.Series(df[col])
                                Xt = imp.fit_transform(X)  # Get a new series
                                df[col] = Xt
                            except ValueError:
                                df.dropna(subset=[col], inplace=True)
                if (df[col].isnull().sum() > 0):
                    df.drop(col, inplace=True, axis=1)
                    feature_names.remove(col)
    features = list(filter(lambda x: x[0] in feature_names, features))
    return df, feature_names, features


def preparewithNames(name, metadata, filename, column=[], target="class", binary=False):
    """
    Parameters
    ----------
    target : str, optional
        Name the variable scope
    Returns
   -------
   df
       a dataframe managed
    [e, f, categorical_names, df_encodes, label_encoder]
        parameter for change categorical in a model
           """

    def one_hot_encoding(df, class_name):
        if not isinstance(class_name, list):
            dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
            class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
            dfY = df[class_name].map(class_name_map)
            df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
            feature_names = dfX.columns
            class_values = sorted(class_name_map)
        else:
            dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
            # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
            class_values = sorted(class_name)
            dfY = df[class_values]
            df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
            feature_names = dfX.columns
        return df, feature_names, class_values

    def get_numeric_columns(df):
        numeric_columns = df._get_numeric_data().columns
        return numeric_columns

    def get_real_feature_names(rdf, numeric_columns, class_name):
        if isinstance(class_name, list):
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
        else:
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
        return real_feature_names

    def get_features_map(feature_names, real_feature_names):
        features_map = defaultdict(dict)
        i = 0
        j = 0

        while i < len(feature_names) and j < len(real_feature_names):
            if feature_names[i] == real_feature_names[j]:
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
                j += 1
            elif feature_names[i].startswith(real_feature_names[j]):
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
            else:
                j += 1
        return features_map

    def get_datetime(args, df):
        for c in args:
            df[c] = pd.to_datetime(df[c])
        return df

    def define_df(df, columns):
        return df[columns]

    def redefine_df(df):
        if (any(df.loc[0, df.columns != 'class'] == [x for x in df.columns if
                                                     x != "class"])):  # se la prima riga è uguale alle mie colonne
            df = pd.read_csv(filename, skipinitialspace=True, delimiter=',', names=df.columns,
                             header=0,
                             usecols=col_indexes)
            h = 0
        else:
            h = 1
            # se ho possibili valori nan
        if (any('?' in df[col].unique() for col in df.columns)):
            df = pd.read_csv(filename, skipinitialspace=True, header=h, names=df.columns,
                             delimiter=',',
                             usecols=col_indexes, na_values='?', keep_default_na=True)
        return df
    feature_names, features, col_indexes = get_features(metadata)
#    f = [a for i, (a, b, c) in enumerate(features)]
 #   target = [x for x in feature_names if x not in f]
    standard = False
    if (name == "compas-scores-two-years" or name == "compas"):
        df = pd.read_csv(filename, skipinitialspace=True, delimiter=',',
                         usecols=col_indexes)
        columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in',
                   'c_jail_out',
                   'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score',
                   'score_text']
        df = define_df(df, columns)
        df = redefine_df(df)
        args = ['c_jail_out', 'c_jail_in']
        df = get_datetime(args, df)
        df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
        df['length_of_stay'] = np.abs(df['length_of_stay'])

        df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
        df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

        df['length_of_stay'] = df['length_of_stay'].astype(int)
        df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
        if binary:
            def get_class(x):
                if x < 7:
                    return 'Medium-Low'
                else:
                    return 'High'

            df['class'] = df['decile_score'].apply(get_class)
        else:
            df['class'] = df['score_text']
        del df['c_jail_in']
        del df['c_jail_out']
        del df['decile_score']
        del df['score_text']
        feature_names = list(df.columns)
    elif (name == "foreveralone"):
        target = "depressed"
        df = pd.read_csv(filename, skipinitialspace=True, delimiter=',',
                         usecols=col_indexes)
        columns = ['gender', 'sexuallity', 'age', 'income', 'race', 'bodyweight', 'virgin', 'prostitution_legal', 'pay_for_sex', 'friends', 'social_fear', 'what_help_from_others','attempt_suicide', 'employment', 'job_title', 'edu_level', 'depressed']
        def get_class(x):
            if x < 15:
                return 'Low'
            else:
                return 'High'
        def what_help(x):
            if(x.startswith("I don't want help")):
                return "I don't want help"
            else:
                return 'Other Help'
        df = define_df(df, columns)
        df['friends'] = df['friends'].apply(get_class)
        df['what_help_from_others'] = df['what_help_from_others'].apply(what_help)
    elif (name == "diabetes"):
        target = "Outcome"
        df = pd.read_csv(filename, skipinitialspace=True, delimiter=',',
                         usecols=col_indexes)
        df = df.drop('DiabetesPedigreeFunction', axis=1)
    else:
        standard = True
    if (standard):
        df = pd.read_csv(filename, skipinitialspace=True, delimiter=',', names=feature_names,
                         usecols=col_indexes)  # .fillna(fill_na).values
    if (target != "class"):
        df['class'] = df[target]
        del df[target]
        target = "class"
        feature_names = list(df.columns)
    df = redefine_df(df)
    df.columns = [c.replace('=', '') for c in df.columns]
    args = ["phone number", 'c_jail_in', 'c_jail_out', 'decile_score', 'score_text', 'education-num', 'fnlwgt']
    df, feature_names, features = NanValues(df, target,feature_names, features, *args)
    numeric_columns = get_numeric_columns(df)
    rdf = df
    df, feature_names, class_values = one_hot_encoding(df, target)
    real_feature_names = get_real_feature_names(rdf, numeric_columns, target)
    rdf = rdf[real_feature_names + (class_values if isinstance(target, list) else [target])]
    features_map = get_features_map(feature_names, real_feature_names)
    model = Tab()
    model.definition(df, rdf, feature_names, class_values, numeric_columns, real_feature_names, features_map)
    return model


def sklearn_metadata(f, fi, fb, n, t):
    res = [0] * (n + 1)
    m = fi[-1]
    idx_fnb = 0  # indice features not binarize
    idx_fb = 0
    i = 0
    class_observed = 0
    for col, col_type, feat_type in f:
        if col == t:
            res[i] = (col, col_type, feat_type)
            i -= 1
            class_observed = 1
        else:
            if i not in fb:
                res[m + idx_fnb + class_observed] = (col, col_type, feat_type)
                idx_fnb += 1
            else:
                for j in range(fi[idx_fb], fi[idx_fb + 1]):
                    res[j + class_observed] = (col, col_type, feat_type)
                idx_fb += 1
        i += 1
    return res


#SIDE OF IMAGE PREPROCESSING

def prepareImage(name,PATH,SHAPE_1,SHAPE_2,dataset,X_vec,y_vec,explainer,label):
    def LoadImage(file_path,SHAPE_1,SHAPE_2):
        im = cv2.imread(file_path)
        im = cv2.resize(im, (SHAPE_1 , SHAPE_2), 3)
        return im / 127.5 - 1.0
    PATH = PATH + ("images/")
    slim = tf.contrib.slim
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    graph = tf.Graph()
    processed_images = tf.placeholder(tf.float32, shape=(None, SHAPE_1, SHAPE_2, 3))
    checkpoints_dir = PATH.replace("images/", "models/")
    img = Img()
    img.name = name
    img.label = label
    if (dataset == "mnist" or dataset == "faces"):
        if (dataset == "faces"):
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                                train_size=0.70)
            img.segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                                train_size=0.55)
            img.segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        img.x = y_test[name]
        img.im = X_test[name]
    else:
        im = LoadImage(PATH + name,SHAPE_1,SHAPE_2)
  #      print("Label: {}".format(img.label))
    if (isinstance(explainer, ex.Lime)):
        if (dataset == "mnist" or dataset== "faces"):
            img.set_pipiline()
            img.simple_rf_pipeline.fit(X_train, y_train)
            img.predict_fn = img.simple_rf_pipeline.predict_proba
        else:
            def transform_img_fn(path_list,SHAPE_1,SHAPE_2):
                out = []
                for f in path_list:
                    with open(f, 'rb') as t:
                        contents = t.read()
                    image_raw = tf.image.decode_jpeg(contents, channels=3)
                    image = inception_preprocessing.preprocess_image(image_raw, SHAPE_1, SHAPE_2,
                                                                     is_training=False)
                out.append(image)
                return session.run([out])[0]
            names = imagenet.create_readable_names_for_imagenet_labels()
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, _ = inception.inception_v3(processed_images, reuse=tf.AUTO_REUSE, num_classes=1001,
                                                       is_training=False)
            probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
                slim.get_model_variables('InceptionV3'))
            session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            init_fn(session)
            bb = kerasinception_v3.InceptionV3()
            images = transform_img_fn([PATH + name],SHAPE_1,SHAPE_2)
            img.definition(bb, session, images, processed_images, probabilities, names)
            img.x = printPrediction(img,True)
            tf.reset_default_graph()
            K.clear_session()
    elif (isinstance(explainer, ex.Lore)):
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
    elif (isinstance(explainer, ex.DeepExplainer)):
        if (dataset == "mnist" or dataset == "faces"):
            if (dataset == "mnist"):
                sys.path.insert(0, os.path.abspath('..'))
                tmp_dir = tempfile.gettempdir()
                mnist = input_data.read_data_sets(tmp_dir, one_hot=True)
                n_hidden_1 = 256  # 1st layer number of neurons
                n_hidden_2 = 256  # 2nd layer number of neurons
                num_input =  SHAPE_1 * SHAPE_2  # MNIST data input (img shape: 28*28)
                num_classes = 10  # MNIST total classes (0-9 digits)
                X = tf.placeholder("float", [None, num_input])
                weights = {
                    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], mean=0.0, stddev=0.05)),
                    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0.0, stddev=0.05)),
                    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], mean=0.0, stddev=0.05))
                }
                biases = {
                    'b1': tf.Variable(tf.zeros([n_hidden_1])),
                    'b2': tf.Variable(tf.zeros([n_hidden_2])),
                    'out': tf.Variable(tf.zeros([num_classes]))
                }

                def model(x, act=tf.nn.relu):
                    layer_1 = act(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
                    layer_2 = act(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
                    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
                    return out_layer
                logits = model(X)
                init = tf.global_variables_initializer()
                session.run(init)
                def input_transform(x):
                    return (x - 0.5) * 2
                test_x = input_transform(mnist.test.images)
                test_y = mnist.test.labels
                xi = test_x[[name]]
                img.im = test_x[name].reshape(SHAPE_1,SHAPE_2)
                yi = test_y[name]
                img.definition(session,logits, yi, X, xi)
        else:
            with DeepExplain(session=session, graph=session.graph) as de:
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    _, end_points = inception.inception_v3(processed_images, reuse=tf.AUTO_REUSE,num_classes=1001, is_training=False)
                logits = end_points['Logits']
                yi = tf.argmax(logits, 1)
                saver = tf.train.Saver(slim.get_model_variables())
                saver.restore(session, checkpoints_dir + "inception_v3.ckpt")
                filenames, xs = load_images(PATH,name,SHAPE_1,SHAPE_2)
                labels = session.run(yi, feed_dict={processed_images: xs})
                img.definition(end_points,session, xs, labels,logits,processed_images,im)
    elif (isinstance(explainer, ex.Saliency)):
        if (dataset == "mnist" or dataset == "faces"):
    #        im = np.stack((img.im,)*3, axis=-1)
            im = img.im
            with graph.as_default():
                processed_images = tf.placeholder(tf.float32, shape=(None, SHAPE_1, SHAPE_2, 3))
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    _, end_points = inception.inception_v3(processed_images,reuse=tf.AUTO_REUSE, is_training=False, num_classes=1001)
                    # Restore the checkpoint
                    sess = tf.Session(graph=graph)
                    saver = tf.train.Saver()
                    saver.restore(sess, checkpoints_dir + "inception_v3.ckpt")
                # Construct the scalar neuron tensor.
                logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
                neuron_selector = tf.placeholder(tf.int32)
                y = logits[0][neuron_selector]
                # Construct tensor for predictions.
                prediction = tf.argmax(logits, 1)
        else:
            with graph.as_default():
                processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
                with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                    _, end_points = inception_v3.inception_v3(processed_images, is_training=False, num_classes=1001)
                    # Restore the checkpoint
                    sess = tf.Session(graph=graph)
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
    img.SHAPE1 = SHAPE_1
    img.SHAPE2 = SHAPE_2
    return img

def load_images(PATH,name,SHAPE_1,SHAPE_2):
    images = np.zeros((1, SHAPE_1 , SHAPE_2, 3))
    filenames = []
    idx = 0
    for filepath in tf.gfile.Glob(os.path.join(PATH, '*.JPEG')):
        if(filepath == name):
            with tf.gfile.Open(filepath, 'rb') as f:
                image = imread(f, mode='RGB').astype(np.float) / 255.0
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0
            filenames.append(os.path.basename(filepath))
            idx += 1
    return filenames, images




