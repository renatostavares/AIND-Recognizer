import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError

        best_score = float('inf')
        best_model = None

        num_features = len(self.X[0])
        logN = np.log(len(self.X))

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

                p = num_states * num_states + 2 * num_states * num_features - 1
                score = -2 * logL + p * logN

            except:
                continue

            if score < best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError

        best_score = float('-inf')
        best_model = None

        M = len((self.words).keys())

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

            except:
                logL = float("-inf")

            log_sum = 0
            for word in self.hwords.keys():
                ix_word, word_lengths = self.hwords[word]

            try:
                log_sum += hmm_model.score(ix_word, word_lengths)

            except:
                log_sum += 0

            score = logL - (1 / (M - 1)) * (log_sum - (0 if logL == float("-inf") else logL))

            if score > best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #raise NotImplementedError

        best_score = float('-inf')
        best_model = None

        if len(self.sequences) < 2:
            return None

        split_method = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            logL_total = 0
            counter = 0

            for train_idx, test_idx in split_method.split(self.sequences):
                X_train, train_length = combine_sequences(train_idx, self.sequences)
                X_test, test_length = combine_sequences(test_idx, self.sequences)

                try:
                    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, train_length)
                    logL = model.score(X_test, test_length)
                    counter += 1
                except:
                    logL = 0

                logL_total += logL

            score = float('-inf')
            if counter != 0:
                score = logL_total / counter

            if score > best_score:
                best_score = score
                best_model = model

        return best_model
