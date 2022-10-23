import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
def read_dataset(file,format=0):
 global file1_name, file2_name
 if file == 'train':
  file1_name = 'train_data.csv'
  file2_name = 'train_datalen.csv'
 elif file == 'test':
  file1_name = 'test_data.csv'
  file2_name = 'test_datalen.csv'
 featueres = 39
 data = pd.read_csv(file1_name, header=None)
 print(len(data))
 X = data.values
 y = X[:, featueres]
 X = X[:, 0:featueres]
 encoder = LabelEncoder()
 encoder.fit(y)
 y_ = encoder.transform(y)
 Y__ = one_hot_encoding(y_)
 data = pd.read_csv(file2_name, header=None)
 Xlen = data.values
 X_order = np.array(range(1, len(Y__) + 1))
 if format == 1:
  Y__ = None
  Y__ = y_
 X, Y__, data_order = shuffle(X, Y__, X_order, random_state=1)
 return X, Y__, data_order, Xlen
def one_hot_encoding(lables):
 n_lables = len(lables)
 n_unique_lables = len(np.unique(lables))
 one_hot_encode = np.zeros((n_lables,n_unique_lables))
 one_hot_encode[np.arange(n_lables),lables] = 1
 return one_hot_encode