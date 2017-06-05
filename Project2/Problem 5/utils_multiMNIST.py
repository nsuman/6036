import gzip, _pickle, numpy as np
from keras.utils import np_utils
num_classes = 10
img_rows, img_cols = 42, 28

def get_data(path_to_data_dir, use_mini_dataset):
	if use_mini_dataset:
		exten = '_mini'
	else:
		exten = ''
	f = gzip.open(path_to_data_dir + 'train_multi_digit' + exten + '.pkl.gz', 'rb')
	X_train = _pickle.load(f, encoding='latin1')
	f.close()
	X_train =  np.reshape(X_train, (len(X_train), 1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'test_multi_digit' + exten +'.pkl.gz', 'rb')
	X_test = _pickle.load(f, encoding='latin1')
	f.close()
	X_test =  np.reshape(X_test, (len(X_test),1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'train_labels' + exten +'.txt.gz','rb')
	y = np.loadtxt(f)
	f.close()
	y_train = [None] * 2
	y_train[0] = np_utils.to_categorical(y[0], num_classes)
	y_train[0] = np.reshape(y_train[0], (len(y_train[0]), num_classes) )
	y_train[1] = np_utils.to_categorical(y[1],num_classes)
	y_train[1] = np.reshape(y_train[1], (len(y_train[1]), num_classes) )
	f = gzip.open(path_to_data_dir +'test_labels' + exten + '.txt.gz', 'rb')
	y = np.loadtxt(f)
	f.close()
	y_test = [None] * 2
	y_test[0] = np_utils.to_categorical(y[0], num_classes)
	y_test[1] = np_utils.to_categorical(y[1],num_classes)
	y_test[0] = np.reshape(y_test[0], (len(y_test[0]), num_classes))
	y_test[1] = np.reshape(y_test[1], (len(y_test[1]), num_classes))
	return X_train, y_train, X_test, y_test

