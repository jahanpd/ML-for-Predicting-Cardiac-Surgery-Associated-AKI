import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, layers=1, nodes=1, dropout=1, epochs=1):
        tf.keras.backend.clear_session()
        self.optimizer = tf.keras.optimizers.Adam() # use the Adam optimizer
        self.loss = tf.keras.losses.CategoricalCrossentropy() # Use Cat Cross Entropy loss
        self.layers = layers
        self.nodes = nodes
        self.dropout = dropout
        self.epochs = epochs


    def _build(self, layers, nodes, dropout):
        inputs = Input(shape=(self.n,))
        f = Dense(nodes, activation=tf.nn.leaky_relu)(inputs)
        for n in np.arange(int(layers)):
            f = Dense(int(nodes), activation=tf.nn.leaky_relu)(f)
            f = Dropout(dropout)(f)
        output = Dense(2, activation='softmax')(f)
        self.mlp = Model(inputs=inputs, outputs=[output])
        self.mlp.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.AUC()])
        # self.mlp.summary()

    def fit(self, X, y, path=None): # define a training function
        X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), y.astype(np.float32), test_size=0.33, random_state=42)
        self.onehot = onehot()
        self.onehot.fit(y.reshape((-1,1)))
        self.m, self.n = X.shape # m is rows (number of samples) and n is cols (features)
        self._build(self.layers, self.nodes, self.dropout)
        # don't need kfold that oversamples because I have a fit method that incorporates it.
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, Y_res = smote.fit_resample(X_train, y_train)
        y_train = self.onehot.transform(Y_res.reshape((-1,1))).toarray()
        y_test = self.onehot.transform(y_test.reshape((-1,1))).toarray()
        es = EarlyStopping(monitor='val_auc', mode='max', patience = 5, verbose=1,restore_best_weights=True)
        self.history = self.mlp.fit(x=X_res,
                                    y=y_train,
                                    validation_data=(X_test,y_test),
                                    epochs=self.epochs,
                                    callbacks=[es],
                                    verbose=0)

    def predict(self, X): # have the option to predict on new data alone, or test set
        predictions = self.mlp.predict(x=X.astype(np.float32))
        print(np.argmax(predictions,axis=1))
        return np.argmax(predictions,axis=1)

    def predict_proba(self, X):
        predictions = self.mlp.predict(x=X.astype(np.float32))
        return predictions

    def get_params(self, deep=True):
        return {'layers':self.layers, 'nodes':self.nodes, 'dropout':self.dropout, 'epochs':self.epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
