import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)


# Get training dataset for ali
def get_train(*args):
    x = np.load('data/backblaze_train_x.npy').astype(np.float32)
    y = np.load('data/backblaze_train_y.npy').astype(np.float32)
    # (99506, 256) (99506,)
    # 98381 : 1125

    ind = [y==0]
    x_train = x[ind]
    y_train = y[ind]

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    
    return x_train, y_train


# Get testing dataset for ali
def get_test(*args):
	# seed shuffling
    rng = np.random.RandomState(42)
    x = np.load('data/backblaze_test_x.npy').astype(np.float32)
    y = np.load('data/backblaze_test_y.npy').astype(np.float32)
    # (24761, 256) (24761,)
    # 24481 : 280
    
    x_major = x[y==0]
    y_major = y[y==0]
    x_minor = x[y==1]
    y_minor = y[y==1]
    
	# contaminate_rate : the empirical ratio of anomalous samples = anomalous / (normal + anomalous)
    contaminate_rate = len(y_minor) / (len(y_minor) + len(y_major))

    size_major = x_major.shape[0]
    inds = rng.permutation(size_major)
    x_major, y_major = x_major[inds], y_major[inds]

    size_minor = x_minor.shape[0]
    inds = rng.permutation(size_minor)
    x_minor, y_minor = x_minor[inds], y_minor[inds]

    x_test = np.concatenate((x_major, x_minor), axis=0)
    y_test = np.concatenate((y_major, y_minor), axis=0)

    size_test = x_test.shape[0]
    inds = rng.permutation(size_test)
    x_test, y_test = x_test[inds], y_test[inds]

    scaler = MinMaxScaler()
    scaler.fit(x_test)
    scaler.transform(x_test)

    return x_test, y_test, contaminate_rate
