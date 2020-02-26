import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from algorithms.custom import Euclidian
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as onehot
from tensorflow.keras.utils import multi_gpu_model


class autoencoder:
    def __init__(self, dropout=1, dim_red=1, epochs=50, path = None):
        tf.keras.backend.clear_session()
        self.path = path

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.activation = tf.nn.leaky_relu

        self.dim_red = dim_red
        self.dropout = dropout
        self.epochs = epochs

    def _build(self, dropout, dim_red):

        # define autoencoder with shared variables
        inputs_all = Input(shape=(self.n,))
        inputs_a, inputs_b, inputs_c = tf.split(inputs_all, num_or_size_splits=3, axis=1)

        # define shared layers
        dense1 = Dense(self.n, activation=self.activation)
        dense2 = Dense(int(self.n*0.5), activation=self.activation)
        dense3 = Dense(int(self.n*0.5), activation=self.activation)
        dense4 = Dense(int(self.n*0.5), activation=self.activation)
        dense5 = Dense(int(self.n*0.5), activation=self.activation)
        encoded = Dense(int(dim_red), activation=self.activation)
        decoded = Dense(self.n, activation=self.activation)

        a, b, c = dense1(inputs_a),dense1(inputs_b),dense1(inputs_c)
        a, b, c = Dropout(dropout)(a),Dropout(dropout)(b),Dropout(dropout)(c)
        a, b, c = dense2(a),dense2(b),dense2(c)
        a, b, c = Dropout(dropout)(a),Dropout(dropout)(b),Dropout(dropout)(c)
        a, b, c = dense3(a),dense3(b),dense3(c)
        a, b, c = Dropout(dropout)(a),Dropout(dropout)(b),Dropout(dropout)(c)
        ea, eb, ec = encoded(a),encoded(b),encoded(c)
        a, b, c = Dropout(dropout)(ea),Dropout(dropout)(eb),Dropout(dropout)(ec)
        a, b, c = dense4(ea),dense4(eb),dense4(ec)
        a, b, c = Dropout(dropout)(a),Dropout(dropout)(b),Dropout(dropout)(c)
        a, b, c = dense5(a),dense5(b),dense5(c)
        a, b, c = Dropout(dropout)(a),Dropout(dropout)(b),Dropout(dropout)(c)
        da, db, dc = decoded(a),decoded(b),decoded(c)

        # define euclidian layer and process
        euclidian_ab = Euclidian(1)([da,db])
        euclidian_ac = Euclidian(1)([da,dc])

        merged = tf.stack([euclidian_ab,euclidian_ac],axis=1)

        # softmax
        output = Softmax()(merged)

        self.autoencode = Model(inputs=inputs_all, outputs=output)
        try:
            self.autoencode = multi_gpu_model(self.autoencode, gpus=2)
            print("Training using multiple GPUs..")
        except Exception as e:
            print("Training using single GPU or CPU..")

            print(e)

        self.autoencode.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y):
        self.onehot = onehot()
        y_oh = self.onehot.fit_transform(y.reshape((-1,1))).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), y_oh.astype(np.float32), test_size=0.1, random_state=42)
        self.m, self.n = X_train.shape # m is rows (number of samples) and n is cols (features)

        lab_train, lab_test = np.full((len(X_train),2),[0,1]), np.full((len(X_test),2),[0,1])

        self._build(self.dropout, self.dim_red)

        es = EarlyStopping(monitor='val_loss',patience=5,verbose=1,restore_best_weights=True)
        self.history = self.autoencode.fit(x=X_train,
                                           y=lab_train,
                                           validation_data=(X_test,lab_test),
                                           epochs=self.epochs,
                                           callbacks=[es])

    def predict(self, X):
        predictions = np.array(()) # array to store predictions
        splits = np.split(X, 3, axis=1)
        x_a,x_b,x_c = splits
        samples = np.split(X[np.random.choice(len(X), 100, replace=True)],3, axis=1)
        temp_a,temp_b,temp_c = samples
        for n in range(len(X)):
            temp = np.array(()) # array to store predictions
            x_same = np.tile(x_a[n],100).reshape((100,-1))
            X_pred = np.concatenate((x_same,temp_b,temp_c), axis=1)
            prediction = self.autoencode.predict(x=X_pred.astype(np.float32))
            temp = np.sum(prediction, axis=0)
            predictions = np.append(predictions, np.argmin(temp))
        return predictions.reshape((-1,1))

    def predict_proba(self, X):
        predictions = np.array(()) # array to store predictions
        splits = np.split(X, 3, axis=1)
        x_a,x_b,x_c = splits
        samples = np.split(X[np.random.choice(len(X), 100, replace=True)],3, axis=1)
        temp_a,temp_b,temp_c = samples
        for n in range(len(X)):
            temp = np.array(()) # array to store predictions
            x_same = np.tile(x_a[n],100).reshape((100,-1))
            X_pred = np.concatenate((x_same,temp_b,temp_c), axis=1)
            prediction = self.autoencode.predict(x=X_pred.astype(np.float32))
            temp = np.mean(prediction, axis=0)
            predictions = np.append(predictions, 1-temp[0])
        return predictions.reshape((-1,1))

    def get_params(self, deep=True):
        return {'dim_red':self.dim_red, 'dropout':self.dropout, 'epochs':self.epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
