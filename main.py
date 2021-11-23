from sklearn import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_rel
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
import numpy as np

dataset = 'thyroid4'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clfs = {
    # Algorytm uczy sieć metodą propagacji wstecznej. Nie trzeba więc tego dodatkowo ustawiać.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    'MLP_A': MLPClassifier(hidden_layer_sizes = (10,), max_iter=400, random_state = 75, solver = 'sgd', momentum = 0),
    'MLP_B': MLPClassifier(hidden_layer_sizes = (80,), max_iter=400, random_state = 75, solver = 'sgd', momentum= 0),
    'MLP_C': MLPClassifier(hidden_layer_sizes = (400,), max_iter=400, random_state = 75, solver = 'sgd', momentum= 0),
    #momentum ma zakres 0-1, domyślnie 0,9
    'MLP_AM': MLPClassifier(hidden_layer_sizes = (10,), max_iter=400, random_state = 75, solver = 'sgd'),
    'MLP_BM': MLPClassifier(hidden_layer_sizes = (80,), max_iter=400, random_state = 75, solver = 'sgd'),
    'MLP_CM': MLPClassifier(hidden_layer_sizes = (400,), max_iter=400, random_state = 75, solver = 'sgd'),
}

preproc = SMOTE(random_state=1000)

values = []
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(clfs), n_splits * n_repeats))

for i in range(1, 8):
    x = X[:,  :i]

    for fold_id, (train, test) in enumerate(rskf.split(x, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            x_train, y_train = preproc.fit_resample(x[train], y[train]) 

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x[test])
            scores[clf_id, fold_id] = balanced_accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
    
    values.append(scores)

np.save('results', values)
# ###########################################################################
# #                        analiza statystyczna                             #
# ###########################################################################
# scores = np.load('results.npy')
# print(scores)

# alfa = .05
# t_statistic = np.zeros((len(clfs), len(clfs)))
# p_value = np.zeros((len(clfs), len(clfs)))

# for i in range(len(clfs)):
#     for j in range(len(clfs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

# headers = ["10 neuronów bez momentum", "80 neuronów bez momentum", "400 neuronów bez momentum", "10 neuronów z momentum", "80 neuronów z momentum", "400 neuronów z momentum"]
# names_column = np.array([["10 neuronów bez momentum"], ["80 neuronów bez momentum"], ["400 neuronów bez momentum"], ["10 neuronów z momentum"], ["80 neuronów z momentum"], ["400 neuronów z momentum"]])
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

# advantage = np.zeros((len(clfs), len(clfs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)

# significance = np.zeros((len(clfs), len(clfs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)

# stat_better = significance * advantage
# stat_better_table = tabulate(np.concatenate(
#     (names_column, stat_better), axis=1), headers)
# print("Statistically significantly better:\n", stat_better_table)


