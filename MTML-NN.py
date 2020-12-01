import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_curve, auc
import joblib

from keras import regularizers
from keras.layers import Dense, Conv1D, Dropout, Input, Flatten, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model
import keras
import tensorflow as tf
from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss

class MTML_NN:
    def __init__(self, featureDimension, CNN = False, task_type = 'reg'):
        self.task_type = task_type
        self.CNN = CNN
        if self.task_type == 'reg':
            self.output_unit = 1
            self.output_activation = 'relu'
            self.loss_func = 'mean_squared_error'
        elif self.task_type == 'cla2':
            self.output_unit = 1
            self.output_activation = 'sigmoid'
            self.loss_func = 'binary_crossentropy'
        elif self.task_type == 'cla3':
            self.output_unit = 3
            self.output_activation = 'softmax'
            self.loss_func = 'categorical_crossentropy'
        self.model = self.demographic_model(featureDimension, CNN)
    
    def binary_focal_loss(self, gamma=2, alpha=0.25):
        """
        Binary form of focal loss.
        focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

            p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
            return K.mean(focal_loss)
        return binary_focal_loss_fixed
    
    def multi_category_focal_loss1(self, alpha=0.25, gamma=2.0):
        """
        focal loss for multi category of multi label problem
        Usage:
         model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        epsilon = 1.e-7
        alpha = tf.constant(alpha, dtype=tf.float32)
        #alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
        #alpha = tf.constant_initializer(alpha)
        gamma = float(gamma)
        def multi_category_focal_loss1_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
            ce = -tf.math.log(y_t)
            weight = tf.pow(tf.subtract(1., y_t), gamma)
            fl = tf.matmul(tf.multiply(weight, ce), alpha)
            loss = tf.reduce_mean(fl)
            return loss
        return multi_category_focal_loss1_fixed
    
    def demographic_model(self, featureDimension, CNN):

        def dense_dropout(units, dropout_rate, last_layer, regularizer=regularizers.l2(0.1), activation='relu'):
            dense = Dense(units, kernel_regularizer=regularizer)(last_layer)
            dropout = Dropout(dropout_rate)(dense)
            activation = Activation(activation)(dropout)
            return activation

        if CNN == True:
            inputShape = (featureDimension, 1)
        else:
            inputShape = (featureDimension, )

        inp = Input(shape=inputShape)
        if CNN == True:
            conv = Conv1D(32, kernel_size=(featureDimension), padding='valid', kernel_regularizer=regularizers.l2(0.1))(inp)
            bn = BatchNormalization()(conv)
            conv_act = Activation('relu')(bn)
            flat = Flatten()(conv_act)
            fl = Dense(128, activation='relu', kernel_initializer='random_uniform')(flat)
        else:
            fl = Dense(128, activation='relu', kernel_initializer='random_uniform')(inp)
        dropout = Dropout(0.5)(fl)

        # for nurses
        dense_nurse = Dense(64, activation='relu', kernel_initializer='random_uniform')(dropout)
        dense_alert_nurse = dense_dropout(32, 0.3, dense_nurse)
        dense_happy_nurse = dense_dropout(32, 0.3, dense_nurse)
        dense_energy_nurse = dense_dropout(32, 0.3, dense_nurse)
        dense_health_nurse = dense_dropout(32, 0.3, dense_nurse)
        dense_relax_nurse = dense_dropout(32, 0.3, dense_nurse)
        outp_alert_nurse = Dense(self.output_unit, activation=self.output_activation)(dense_alert_nurse)
        outp_happy_nurse = Dense(self.output_unit, activation=self.output_activation)(dense_happy_nurse)
        outp_energy_nurse = Dense(self.output_unit, activation=self.output_activation)(dense_energy_nurse)
        outp_health_nurse = Dense(self.output_unit, activation=self.output_activation)(dense_health_nurse)
        outp_relax_nurse = Dense(self.output_unit, activation=self.output_activation)(dense_relax_nurse)

        # for doctors
        dense_doctor = Dense(64, activation='relu', kernel_initializer='random_uniform')(dropout)
        dense_alert_doctor = dense_dropout(32, 0.3, dense_doctor)
        dense_happy_doctor = dense_dropout(32, 0.3, dense_doctor)
        dense_energy_doctor = dense_dropout(32, 0.3, dense_doctor)
        dense_health_doctor = dense_dropout(32, 0.3, dense_doctor)
        dense_relax_doctor = dense_dropout(32, 0.3, dense_doctor)
        outp_alert_doctor = Dense(self.output_unit, activation=self.output_activation)(dense_alert_doctor)
        outp_happy_doctor = Dense(self.output_unit, activation=self.output_activation)(dense_happy_doctor)
        outp_energy_doctor = Dense(self.output_unit, activation=self.output_activation)(dense_energy_doctor)
        outp_health_doctor = Dense(self.output_unit, activation=self.output_activation)(dense_health_doctor)
        outp_relax_doctor = Dense(self.output_unit, activation=self.output_activation)(dense_relax_doctor)

        model = Model(inputs=inp, outputs=[outp_alert_nurse, outp_happy_nurse, outp_energy_nurse, outp_health_nurse,
                                           outp_relax_nurse,
                                           outp_alert_doctor, outp_happy_doctor, outp_energy_doctor, outp_health_doctor,
                                           outp_relax_doctor])

        model.compile(loss=self.loss_func, optimizer='adam')
        return model


    def cal_Labels_SampleWeight(self, table, num_label=5):
        '''
        Input: table is a array that have 5 wellbeing labels as the first 5 columns;
            the last column should be the job role information "N" or "D"
        '''
        labels, array = table[:, :5], table[:, -1]
        sample_weights = np.zeros((len(array), num_label * 2))
        for i in range(len(array)):
            if array[i] == 'N':
                sample_weights[i, :] = np.array([1] * num_label + [0] * num_label)
            elif array[i] == 'D':
                sample_weights[i, :] = np.array([0] * num_label + [1] * num_label)

        converted_labels = np.zeros((len(labels), num_label * 2))
        for i in range(len(array)):
            if array[i] == 'N':
                converted_labels[i, :] = np.array(list(labels[i, :]) + [0] * num_label)
            elif array[i] == 'D':
                converted_labels[i, :] = np.array([0] * num_label + list(labels[i, :]))
        return [converted_labels[:, x] for x in range(num_label * 2)], [sample_weights[:, x] for x in range(num_label * 2)]


    def cal_test_SampleWeight(self, job_role, num_label=5):
        '''
        Input: job role information "N" or "D"
        '''
        if type(job_role) == np.ndarray or list:
            sample_weights = np.zeros((len(job_role), num_label * 2))
            for i in range(len(job_role)):
                if job_role[i] == 'N':
                    sample_weights[i, :] = np.array([1] * num_label + [0] * num_label)
                elif job_role[i] == 'D':
                    sample_weights[i, :] = np.array([0] * num_label + [1] * num_label)
        elif type(job_role) == str:
            sample_weights = np.zeros((1, num_label * 2))
            if job_role == 'N':
                sample_weights[0, :] = np.array([1] * num_label + [0] * num_label)
            elif job_role == 'D':
                sample_weights[0, :] = np.array([0] * num_label + [1] * num_label)
        else:
            print("Please enter job role as string or list.")

        return [sample_weights[:, x] for x in range(num_label * 2)]

    def getTrueLabel(self, codedLabel, sample_weight):
        decodedLabel = np.array(codedLabel).T
        if len(decodedLabel.shape) == 3 and decodedLabel.shape[0] == 1:
            decodedLabel = decodedLabel[0]
        sample_weight_decoded = np.array(sample_weight).T
        res = np.zeros((len(decodedLabel), 5))
        for i in range(len(decodedLabel)):
            col_index = sample_weight_decoded[i, :] == 1
            res[i, :] = decodedLabel[i, col_index]
        return res

    def train(self, X_train, y_train, epoch=80, batch_size=16):
        self.normalizer = MinMaxScaler()
        X_train = self.normalizer.fit_transform(X_train)
        normalizer_dir = "./models/normalizer.save"
        joblib.dump(self.normalizer, normalizer_dir)
        
        if self.task_type == 'cla2':
            y_train[:,:-1] = y_train[:,:-1] // 50.01
        if self.task_type == 'cla3':
            y_train[:,:-1] = y_train[:,:-1] // 33.34
            
        y_train, sample_weights = self.cal_Labels_SampleWeight(y_train)

        if self.CNN:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        if self.task_type == 'cla3':
            y_train_cate = []
            for yt in y_train:
                y_train_cate.append(to_categorical(yt))
            y_train = y_train_cate

        self.model.fit(X_train, y_train,
                       epochs=epoch,
                       batch_size=batch_size,
                       sample_weight=sample_weights,
                       verbose=False)
        self.model.save('./models/mtml_nn_'+ self.task_type + '.h5')
        print('Training complete!')

    def predict(self, X, job_role):
        model = load_model('./models/mtml_nn_'+ self.task_type + '.h5')
        sample_weights_test = self.cal_test_SampleWeight(job_role)
        try:
            normalizer = self.normalizer
        except:
            normalizer_filename = "./models/normalizer.save"
            normalizer = joblib.load(normalizer_filename) 
        X_test = normalizer.transform(X)

        if self.CNN:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        if self.task_type == 'cla3':
            y_pred_true = self.getTrueLabel(np.argmax(np.round(model.predict(X_test)), axis=2), sample_weights_test)
        elif self.task_type == 'cla2':
            y_pred_true = self.getTrueLabel(np.round(model.predict(X_test)), sample_weights_test)
        elif self.task_type == 'reg':
            y_pred_true = self.getTrueLabel(model.predict(X_test), sample_weights_test)

        return y_pred_true
