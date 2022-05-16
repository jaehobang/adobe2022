"""
Research Question 1.
Looking at the HDFS dataset (11 million rows), 570k labeled blockIDs,
it could be interesting to localize exactly which words contributed to defining a row as an anomaly or not.

To do this, it will involve a lot of parsing, and experiments.
Let's write the util functions here


"""

import sys
sys.path.append('/nethome/jbang36/adobe_internship')

from logparser.logparser.Drain import Drain
import pandas as pd
import os
import torch
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from lime.lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime.lime_text import LimeTextExplainer


def load_labels(data_location):
    labels = pd.read_csv(os.path.join(data_location, 'anomaly_label.csv'))
    return labels

def preprocess_drain(data_location):
    input_dir  = data_location  # The input directory of log file
    output_dir = 'results/'  # The output directory of parsing results
    log_file   = 'HDFS.log'  # The input log file name
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    # Regular expression list for optional preprocessing (default: [])
    regex      = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]
    st         = 0.5  # Similarity threshold
    depth      = 4  # Depth of all leaf nodes

    template = os.path.join(output_dir, 'HDFS.log_templates.csv')
    structured = os.path.join(output_dir, 'HDFS.log_structured.csv')

    if not os.path.isfile(template):

        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
        parser.parse(log_file)


    template_pandas = pd.read_csv(template)
    structured_pandas = pd.read_csv(structured)

    return template_pandas, structured_pandas


def restructure_data(structured_pd, labels_pd):
    ### we create a dict where each blockId is a key.
    block_dict = {key: [] for key in labels_pd['BlockId']}
    labels_dict = {label['BlockId']: label['Label'] for index, label in labels_pd.iterrows()}

    exceptions = []
    for i, row in tqdm(structured_pd.iterrows(), desc='Appending Relevant Rows'):
        parameter_list = row['ParameterList']
        parameter_list = eval(parameter_list)
        for parameter in parameter_list:
            if 'blk' in parameter:
                if parameter in block_dict:
                    block_dict[parameter].append(i)
                else:
                    ## we need to split by space and then perform parsing again
                    parameter_ = parameter.split(' ')
                    for p in parameter_:
                        if 'blk' in p and p in block_dict:
                            block_dict[p].append(i)
                        elif 'blk' in p and p not in block_dict:
                            exceptions.append(p)

    X = []
    y = []

    for block, relevant_rows in tqdm(block_dict.items(), desc='Generating X,y'):
        relevant_logs = list(structured_pd['EventTemplate'][relevant_rows])
        X.append(" ".join(relevant_logs))
        if labels_dict[block] == 'Normal':
            y.append(0)
        else:
            assert(labels_dict[block] == 'Anomaly')
            y.append(1)

    return X,y

def transform_data(X,y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (X_train, y_train), (X_test, y_test)



def prepare_data(train, test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    X_train, y_train = train
    X_test, y_test = test
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    return (train_vectors, y_train), (test_vectors, y_test), vectorizer

def train_model(X_train,y_train, X_test, y_test):
    class_weight = 'balanced'

    classifier = LogisticRegression(class_weight=class_weight)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    print('f1-score', sklearn.metrics.f1_score(y_test, pred))
    print('precision', sklearn.metrics.precision_score(y_test, pred))
    print('recall', sklearn.metrics.recall_score(y_test, pred))

    return classifier

def init_lime(vectorizer, model):
    c = make_pipeline(vectorizer, model)

    class_names = ['not anomaly', 'anomaly']
    explainer = LimeTextExplainer(class_names=class_names)

    return explainer, c

def get_anomaly_indices(test_labels):
    anomaly_indices = np.where(test_labels == 1)[0]
    return anomaly_indices

def run_lime(ex, c, test_data, test_labels, idx):
    class DataPackage:
        def __init__(self, data, target):
            self.data = data
            self.target = target

    package = DataPackage(test_data, test_labels)
    exp = ex.explain_instance(package.data[idx], c.predict_proba,
                                     num_features=6)  ### select the pipeline, select the datapoint (original), select number of features that it will use for explanation

    exp.show_in_notebook(text=True)

#### okay, now what we need to do after preping the data is running the model... let's run svm?


def save_data(X,y, cache_directory):
    #cache_directory = '/nethome/jbang36/adobe_internship/cache'
    package = {'data': X, 'labels': y}
    data_save_dir = os.path.join(cache_directory, 'HDFS_parsed.json')
    torch.save(package, data_save_dir)

def load_data(cache_directory):
    data_save_dir = os.path.join(cache_directory, 'HDFS_parsed.json')
    package = torch.load(data_save_dir)
    return package['data'], package['labels']


if __name__ == "__main__":
    data_location = '/nethome/jbang36/adobe_internship/data/HDFS_1'

    template_pd, structured_pd = preprocess_drain(data_location)
    labels_pd = load_labels(data_location)

    X, y = restructure_data(structured_pd, labels_pd)

