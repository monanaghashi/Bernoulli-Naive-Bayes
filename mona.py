import re
from math import log
import glob
from collections import Counter


def get_features(text):
    """Extracts features from text

    Args:
        text (str): A blob of unstructured text
    """
    return set([w.lower() for w in text.split(" ")])


class BernoulliNBClassifier(object):

    def __init__(self):
        self._log_priors = None
        self._cond_probs = None
        self.features = None
        

    def train(self, docs, labels):
        """Train a Bernoulli naive Bayes classifier

        Args:
            documents (list): Each element in this list
                is a blog of text
            labels (list): The ground truth label for
                each document
        """

        """Compute log( P(Y) )
        """
        label_counts = Counter(labels)
        N = float(sum(label_counts.values()))
        self._log_priors = {k: log(v/N) for k, v in label_counts.items()}

        # Extract features from each document
        X = [set(get_features(d)) for d in docs]
        # Obtain all features
        self.features = set([f for features in X for f in features])

        """Compute log( P(X|Y) )

           Use Laplace smoothing
           n1 + 1 / (n1 + n2 + 2)
        """
        self._cond_probs = {l: {f: 0. for f in self.features} for l in self._log_priors}

        # Step through each document
        for x, l in zip(X, labels):
            for f in x:
                self._cond_probs[l][f] += 1.

        # Now, compute log probs
        for l in self._cond_probs:
            N = label_counts[l]
            self._cond_probs[l] = {f: (v + 1.) / (N + 2.) for f, v in self._cond_probs[l].items()}

    def predict(self, text):
        """Make a prediction from text
        """
        
        x = get_features(text)
        pred_class = None
        max_ = float("-inf")

        # Perform MAP estimation
        for l in self._log_priors:
            log_sum = self._log_priors[l]
            for f in self.features:
                prob = self._cond_probs[l][f]
                log_sum += log(prob if f in x else 1. - prob)
            if log_sum > max_:
                max_ = log_sum
                pred_class = l

        return pred_class


###Read and pre-process data (clean stop words and convert to lower cases)
reviews_train = []
reviews_test = []
labels_train = []
labels_test = []

file_list = glob.glob('./train/pos/*.txt')
for file_path in file_list:
    with open(file_path, errors='ignore') as f_input:
        reviews_train.append(f_input.read())
        labels_train.append('pos')

file_list = glob.glob('./train/neg/*.txt')
for file_path in file_list:
    with open(file_path, errors='ignore') as f_input:
        reviews_train.append(f_input.read())
        labels_train.append('neg')

myCo = 0
file_list = glob.glob('./test/*.txt')
for file_path in file_list:
    with open(file_path, errors='ignore') as f_input:
        reviews_test.append(f_input.read())
        
        if myCo <= 12499:
            labels_test.append('pos')
        else:
            labels_test.append('neg')
        myCo += 1
                


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


reviews_train = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews_train]
reviews_train = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews_train]

reviews_test = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews_test]
reviews_test = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews_test]


# Train model
print('Number of training examples: {0}'.format(len(labels_train)))
print('Number of test examples: {0}'.format(len(labels_test)))
print('Training begins ...')
nb = BernoulliNBClassifier()
nb.train(reviews_train, labels_train)
print('Training complete!')
print('Number of features found: {0}'.format(len(nb.features)))


# Simple error test metric
print('Testing model...')
f = lambda doc, l: 1. if nb.predict(doc) != l else 0.
num_missed = sum([f(doc, l) for doc, l in zip(reviews_test, labels_test)])

N = len(labels_test) * 1.
error_rate = round(100. * (num_missed / N), 3)

print('Error rate of {0}% ({1}/{2})'.format(error_rate, int(num_missed), int(N)))
