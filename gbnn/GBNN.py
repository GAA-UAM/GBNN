""" Gradient Boosted Neural Network """

# Author: Seyedsaman Emami 
# Author: Gonzalo Martínez-Muñoz 

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)
import copy
import time
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from scipy.special import logsumexp
from sklearn.base import is_classifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.random import sample_without_replacement

class loss:
    """ Template class for the loss function """

    def model0(self, y):
        pass

    def derive(self, y, prev):
        pass

    def newton_step(self, y, residuals, new_predictions):
        pass

    def __call__(self, y, pred):
        pass


class squared_loss(loss):
    """ Squared loss for regression problems """

    def model0(self, y):
        return np.ones(1)*np.mean(y)

    def derive(self, y, prev):
        return y-prev

    def newton_step(self, y, residuals, new_predictions):
        return 1
    
    def __call__(self, y, pred):
        return (y-pred)**2


class classification_loss(loss):
    """ Base class for classification losses """

    def raw_predictions_to_probs(self, preds):
        pass
    

class log_exponential_loss(classification_loss):
    """ Log-exponential loss for binary classification tasks """

    def model0(self, y):
        ymed = np.mean(y)
        return np.ones(1)*(0.5 * np.log((1+ymed) / (1-ymed)))

    def derive(self, y, prev):
        return np.nan_to_num(2.0 * y / (1 + np.exp(2.0 * y * prev)))

    def newton_step(self, y, residuals, new_predictions):
        f_m = new_predictions
        return np.sum(residuals * f_m) / np.sum(residuals * f_m * f_m * (2.0 * y - residuals))

    def raw_predictions_to_probs(self, preds):
        preds = 1 / (1 + np.exp(-2 * preds))
        return np.vstack((1-preds, preds)).T

    def __call__(self, y, pred):
        return np.log(1 + np.exp(-2.0 * y * pred))


class multi_class_loss(classification_loss):
    """ Entropy loss por multi-class classification tasks """

    def model0(self, y):
        return np.zeros_like(y[0, :])

    def derive(self, y, prev):
        return y - np.nan_to_num((np.exp(prev -
                                         logsumexp(prev, axis=1, keepdims=True))))

    def newton_step(self, y, residuals, new_predictions):
        f_m = new_predictions
        p = y-residuals
        return -np.sum(f_m * (y - p)) / np.sum(f_m * f_m * p * (p - 1))

    def raw_predictions_to_probs(self, preds):
        return np.exp(preds - logsumexp(preds, axis=1, keepdims=True))

    def __call__(self, y, pred):
        return np.sum(-1 * (y * pred).sum(axis=1) +
                      logsumexp(pred, axis=1))


class GNEGNE(BaseEstimator):
    """ Base class for gradient booted neural networks """
    
    def __init__(self, loss, total_nn=200, num_nn_step=1, eta=1.0,
                 solver='lbfgs', subsample=0.5, tol=0.0,
                 max_iter=200, random_state=None, activation='logistic'):

        self.loss        = loss
        self.total_nn    = total_nn
        self.num_nn_step = num_nn_step
        self.eta         = eta
        self.solver      = solver
        self.subsample   = subsample
        self.tol         = tol
        self.max_iter    = max_iter
        self.random_state= random_state
        self.activation  = activation

    def print_out(self):
        return str(self.eta) + "_" + str(self.num_nn_step) + "_" + str(self.max_iter) + "_" + str(self.subsample)

    def _add(self, model, step):
        self.models.append(model)
        self.steps.append(step)

    def fit(self, X, y):
        self.T        = int(self.total_nn/self.num_nn_step)
        self.total_nn = self.T * self.num_nn_step

        if is_classifier(self):
            # Create the complete MLP structure
            self.NN = MLPClassifier(hidden_layer_sizes=(self.total_nn,),
                                       max_iter=2,
                                       activation=self.activation,
                                       solver=self.solver,
                                       tol=self.tol)
            self.NN.fit(X, y)
            
            if type_of_target(y) == 'multiclass':
                self.loss  = multi_class_loss()    
                lb = LabelBinarizer()
                y = lb.fit_transform(y)
            else:
                self.loss  = log_exponential_loss()
                lb = LabelBinarizer(pos_label=1, neg_label=-1)
                y = lb.fit_transform(y).ravel()
            self.classes_ = lb.classes_
        else:
            # Create the complete MLP structure
            self.NN = MLPRegressor(hidden_layer_sizes=(self.total_nn,),
                                   max_iter=2,
                                   activation=self.activation,
                                   solver=self.solver,
                                   tol=self.tol)
            self.NN.fit(X, y)

        self.models = []
        self.steps = []

        self.intercept = self.loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        self.losses = []

        random_state = check_random_state(self.random_state)
        self._training_time = []
        t0 = time.time()
        
        for i in range(self.T):
            
            residuals = self.loss.derive(y, acum)
            
            rr = MLPRegressor(hidden_layer_sizes = (self.num_nn_step,),
                              max_iter           = self.max_iter,
                              activation         = self.activation,
                              solver             = self.solver,
                              tol                = self.tol,
                              random_state       = random_state,
                              n_iter_no_change   = self.max_iter)
            
            if self.subsample < 1.0:
                indices = sample_without_replacement(X.shape[0],
                                                     int(self.subsample*X.shape[0]))
                X_i = X[indices]
                residuals_i = residuals[indices]
            else:
                X_i = X
                residuals_i = residuals

            rr.fit(X_i, residuals_i)

            predictions_i = rr.predict(X)

            rho = self.eta * self.loss.newton_step(y, residuals, predictions_i)

            acum = acum + rho * predictions_i
            self.losses.append(np.mean(self.loss(y, acum)))
            self._add(rr, rho)
            self._training_time.append(time.time() - t0)

    def _decision_function(self, X):
        """ outputs the raw prediction of the model """
        preds = self.models[0].predict(X) * self.steps[0] + self.intercept

        for model, step in zip(self.models[1:], self.steps[1:]):
            preds += model.predict(X) * step

        return preds

    def ave_losses(self):
        return self.losses

    def to_NN(self):
        """ Assembles the model into a single NN """
        NN = copy.deepcopy(self.NN)

        multiplier = 2.0 if type(self.loss) is log_exponential_loss else 1.0

        n = self.num_nn_step

        # Input to hidden layer
        for i, model in enumerate(self.models):
            NN.coefs_[0][:, i*n:(i+1)*n] = model.coefs_[0]
            NN.intercepts_[0][i*n:(i+1)*n] = model.intercepts_[0][0:n]

        # Hidden to output layer
        NN.intercepts_[1][:] = np.ones((1,)) * self.intercept * multiplier

        for i, model, rho in zip(range(len(self.models)), self.models, self.steps):
            NN.coefs_[1][i*n:(i+1)*n] = model.coefs_[1] * rho * multiplier
            NN.intercepts_[1][:] += model.intercepts_[1][:] * rho * multiplier

        return NN


class GNEGNEClassifier(GNEGNE, ClassifierMixin):
    """ Gradient booted neural network clasifier """

    def __init__(self, total_nn=200, num_nn_step=1, eta=1.0,
                 solver='lbfgs', subsample=0.5, tol=0.0,  # 1e-4,
                 max_iter=200, random_state=None, activation='logistic'):

        self._estimator_type = 'classifier'

        super(GNEGNEClassifier, self).__init__(multi_class_loss(), total_nn,
                                               num_nn_step, eta, solver, subsample, tol, max_iter,
                                               random_state, activation)

    def _predict(self, probs):
        return self.classes_.take(np.argmax(probs, axis=1))

    def predict(self, X):
        preds = self.predict_proba(X)
        return self._predict(preds)

    def staged_score(self, X, y):
        scores = []

        preds = np.ones_like(self.models[0].predict(X)) * self.intercept

        for model, step in zip(self.models, self.steps):
            preds += model.predict(X) * step
            scores.append(np.mean(self._predict(
                self.loss.raw_predictions_to_probs(preds)) == y))

        return scores

    def predict_proba(self, X):
        return self.loss.raw_predictions_to_probs(self._decision_function(X))


class GNEGNERegressor(GNEGNE, RegressorMixin):
    """ Gradient booted neural network regressor """

    def __init__(self, total_nn=200, num_nn_step=1, eta=1.0,
                 solver='lbfgs', subsample=0.5, tol=0.0,
                 max_iter=200, random_state=None, activation='logistic'):

        self._estimator_type = 'regressor'
        super(GNEGNERegressor, self).__init__(squared_loss(), total_nn,
                                              num_nn_step, eta, solver, subsample, tol, max_iter,
                                              random_state, activation)

    def predict(self, X):
        return self._decision_function(X)

    def staged_score(self, X, y):
        scores = []

        preds = np.ones(X.shape[0])*self.intercept

        for model, step in zip(self.models, self.steps):
            preds += model.predict(X) * step
            scores.append(np.mean((preds-y)**2))

        return scores
