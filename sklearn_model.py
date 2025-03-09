# Codegrade Tag Question1
# Do *not* remove the tag above
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

binary_target = df.columns[0]  # The first column is the target variable

binary_features = [col for col in df.columns if set(df[col].unique()).issubset({0, 1}) and col != binary_target]

y = df[binary_target].values
X = df[binary_features].values


def train_test_split_manual(X, y, train_size=0.75, random_state=None):
    """
    Perform a manual train-test split on the dataset.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)
    else:
        rng = np.random.default_rng()

    shuffled_indices = rng.permutation(indices)

    n_train = int(train_size * n_samples)

    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split_manual(X, y)


class NaiveBinaryBayes:
    """
    Naive Bayes classifier for binary features and binary class labels.

    Matches the interface of sklearn.naive_bayes.CategoricalNB
    """

    def __init__(self, alpha=1.0):
        """
        Construct the classifier object.

        Parameters:
        - alpha : Parameter for Laplace smoothing (how many pseudoinstances to add for each category)
        """
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the dataset X with correct class labels y into the classifier.
        """
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]

        self.classes_, self.class_count_ = np.unique(y, return_counts=True)

        self.class_log_prior_ = np.log(self.class_count_ / y.shape[0])

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.category_count_ = []
        self.feature_log_prob_ = []

        for i in range(n_features):
            feature_counts = np.zeros((len(self.classes_), 2))
            for j, c in enumerate(self.classes_):
                X_c = X[y == c, i]
                feature_counts[j, 0] = np.sum(X_c == 0) + self.alpha
                feature_counts[j, 1] = np.sum(X_c == 1) + self.alpha
            self.category_count_.append(feature_counts)

            log_probs = np.log(feature_counts / (self.class_count_[:, np.newaxis] + 2 * self.alpha))
            self.feature_log_prob_.append(log_probs)

        return self

    def predict_log_proba(self, X):
        """
        Given an m*d array X, returns an m*2 array of log probabilities corresponding to the posterior probability of the observation coming from a given class.

        Parameters:
        - X: an m*d array of observations to predict
        """
        assert X.ndim == 2 and X.shape[1] == self.n_features_in_

        log_prob = np.zeros((X.shape[0], len(self.classes_)))

        for i, c in enumerate(self.classes_):
            log_prob[:, i] = self.class_log_prior_[i]

            for j in range(self.n_features_in_):
                feature_value = X[:, j]

                log_prob[:, i] += (feature_value * self.feature_log_prob_[j][i, 1] + (1 - feature_value) * self.feature_log_prob_[j][i, 0])

        log_prob_max = np.max(log_prob, axis=1, keepdims=True)
        log_prob -= log_prob_max
        log_prob -= np.log(np.sum(np.exp(log_prob), axis=1, keepdims=True))

        return log_prob

    def predict_proba(self, X):
        log_prob = self.predict_log_proba(X)
        prob = np.exp(log_prob)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return self.classes_[np.argmax(prob, axis=1)]


def confusion_matrix_manual(y_true, y_pred):
    conf_mtx = np.zeros((2, 2), dtype=int)
    for observed_val, estimated_val in zip(y_true, y_pred):
        conf_mtx[int(observed_val), int(estimated_val)] += 1
    return conf_mtx

y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
conf_mtx = confusion_matrix_manual(y_true, y_pred)
print("Confusion Matrix:")
print(conf_mtx)


def recall_score_manual(confusion_matrix):
    f_negative = confusion_matrix[1, 0]  # f_neg (actual 1, predicted 0)
    t_pos = confusion_matrix[1, 1]  # t_pos (actual 1, predicted 1)
    return t_pos / (t_pos + f_negative) if (t_pos + f_negative) else 0.0

def precision_score_manual(confusion_matrix):
    f_pos = confusion_matrix[0, 1]  # f_pos(actual 0, predicted 1)
    t_pos = confusion_matrix[1, 1]
    return t_pos / (t_pos + f_pos) if (t_pos + f_pos) else 0.0

def accuracy_score_manual(confusion_matrix):
    t_neg = confusion_matrix[0, 0]  # t_neg (actual 0, predicted 0)
    t_pos = confusion_matrix[1, 1]
    final_val = confusion_matrix.sum()
    return (t_pos + t_neg) / final_val if final_val else 0.0

def f1_score_manual(confusion_matrix):
    precision = precision_score_manual(confusion_matrix)
    recall = recall_score_manual(confusion_matrix)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

mtx_analysis_ex = np.array([[21, 7], [7, 21]])
print("F1-score:", f1_score_manual(mtx_analysis_ex))
print("Precision:", precision_score_manual(mtx_analysis_ex))
print("Recall:", recall_score_manual(mtx_analysis_ex))
print("Accuracy:", accuracy_score_manual(mtx_analysis_ex))


health_frameDATA = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')  # dataset
data_in_x = health_frameDATA.drop('Diabetes_binary', axis=1)
data_in_y = health_frameDATA['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(data_in_x, data_in_y, test_size=0.25, random_state=16445)
nb_category = CategoricalNB(alpha=1)
nb_category.fit(X_train, y_train)
lp_pred = nb_category.predict_log_proba(X_test)
y_pred = nb_category.predict(X_test)
estimated_log = nb_category.predict_log_proba(X_test)
fscore = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("F1score:", fscore)
print("Recall:", recall)
print("Precision:", precision)
print("Accuracy:", accuracy)


val_f_pos, val_t_pos, _ = roc_curve(y_test, estimated_log[:, 1])
fig_roc, ax = plt.subplots()
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.plot(val_f_pos, val_t_pos, label='ROC Curve')
roc_auc = roc_auc_score(y_test, estimated_log[:, 1])
ax.plot([0, 1], [0, 1], linestyle=(0, (7, 11)), color=np.random.rand(3,), linewidth=np.random.uniform(2.5, 4.0), label='Random Classifier')
ax.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
ax.legend()
plt.show()
