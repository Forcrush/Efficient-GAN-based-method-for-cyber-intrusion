import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)

# contaminate_rate : the empirical ratio of anomalous samples = anomalous / (normal + anomalous)
contaminate_rate = 0.2


# Get training dataset for KDD 10 percent
def get_train(*args):
    return _get_adapted_dataset("train")


# Get testing dataset for KDD 10 percent
def get_test(*args):
    return _get_adapted_dataset("test")


# Get shape of the dataset for KDD 10 percent
def get_shape_input():
    return (None, 121)


# Get shape of the labels in KDD 10 percent
def get_shape_label():
    return (None,)


# Gets the basic dataset
def _get_dataset():
    """ 
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 121)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 121)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    
    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # (198361, 121) (198361,) (247011, 121) (247011,)

    return dataset


# Gets the adapted dataset for the experiments
def _get_adapted_dataset(split):
    """ 
    Args :
            split (str): train or test
                         train needn't consider the ratio of anomalous/normal
                         test shoule consider it
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl],
                                                    contaminate_rate)

    return dataset[key_img], dataset[key_lbl]


# Encode text values to dummy variables
def _encode_text_dummy(df, name):
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Converts a Pandas dataframe to the x,y inputs that TensorFlow needs
def _to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)


# Column names of the dataframe
def _col_names():
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


# Adapt the ratio of normal/anomalous data
def _adapt(x, y, contaminate_rate):
    """
    As definited before: 'normal'--1(minor)--anomalous data | '~normal'--0(major)--normal data
    contaminate_rate is aimed to satisy the factual ratio of anomalous data in test dataset
    """
    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test * contaminate_rate / (1 - contaminate_rate))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]
    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy
