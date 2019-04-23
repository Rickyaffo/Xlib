from collections import defaultdict
import pandas as pd
from sklearn_pandas import CategoricalImputer
from scipy.io import arff
import numpy as np
from skmultilearn.dataset import load_from_arff

from algorithms.Models import Tab


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
        else:  # isinstance(class_name, list)
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

    def get_numeric(df, num, class_name):
        for col in df.columns[-num:]:
            df[col] = df[col].apply(pd.to_numeric)
        cols_Y = [col for col in df.columns if col.startswith(class_name)]
        return df, cols_Y

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
    if (len(feature_names) == 0):
        return
    f = [a for i, (a, b, c) in enumerate(features)]
    target = [x for x in feature_names if x not in f]
    if (len(f) == 0):
        target = "class"
    else:
        target = target[0]
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
    elif (name == "yeast"):
        standard = False
        df = pd.DataFrame(arff.loadarff(name + ".arff")[0])
        df, cols_Y = get_numeric(df, 14, target)
    elif (name == "medical"):
        data = load_from_arff(name + ".arff", label_count=45, load_sparse=False, return_attribute_definitions=True)
        cols_X = [i[0] for i in data[2]]
        cols_Y = [i[0] for i in data[3]]
        X_med_df = pd.DataFrame(data[0].todense(), columns=cols_X)
        y_med_df = pd.DataFrame(data[1].todense(), columns=cols_Y)
        df = pd.concat([X_med_df, y_med_df], 1)
    else:
        standard = True
    if (standard):
        df = pd.read_csv(filename, skipinitialspace=True, delimiter=',', names=feature_names,
                         usecols=col_indexes)  # .fillna(fill_na).values
    df = redefine_df(df)
    df.columns = [c.replace('=', '') for c in df.columns]
    if (target != "class"):
        df['class'] = df[target]
        del df[target]  # todo memozziare classe scope
    args = ["phone number", 'c_jail_in', 'c_jail_out', 'decile_score', 'score_text', 'education-num', 'fnlwgt']
    # ci sono delle colonne da eliminare
    #      if (len(column) > 0):
    #        df = self.redefineModel(df, column)
    df, feature_names, features = NanValues(df, target,feature_names, features, *args)  ##todo funzionalità ui
    # prepare dataset
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