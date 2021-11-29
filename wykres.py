# Importing the matplotlib library
import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)

def save_plot_2(with_momentum, without_momentum, title, name):
    plt.figure(figsize=[15, 10])
    X = np.arange(len(with_momentum))
    plt.bar(X, with_momentum, color = 'y', width = 0.33)
    plt.bar(X + 0.33, without_momentum, color = 'b', width = 0.33)
    plt.legend(['Momentum (0.9)', 'Bez momentum'])
    plt.xticks([i + 0.165 for i in range(7)], range(1, 8))
    plt.title(title)
    plt.xlabel('Liczba cech')
    plt.ylabel('Dokładność')
    plt.savefig(name + '.pdf', bbox_inches='tight')

def save_plot_3(m1, m2, m3, title, name):
    plt.figure(figsize=[15, 10])
    X = np.arange(len(m1))
    plt.bar(X, m1, color = 'r', width = 0.25)
    plt.bar(X + 0.25, m2, color = 'g', width = 0.25)
    plt.bar(X + 0.5, m3, color = 'b', width = 0.25)
    plt.legend(['10 neuronów', '80 neuronów', '400 neuronów'])
    plt.xticks([i + 0.25 for i in range(7)], range(1, 8))
    plt.title(title)
    plt.xlabel('Liczba cech')
    plt.ylabel('Dokładność')
    plt.savefig(name + '.pdf', bbox_inches='tight')

values = np.load('results.npy')

# w = [np.mean(i) for i in values[:, 0]]
# o = [np.mean(i) for i in values[:, 3]]
# save_plot_2(w, o, "MLP - 1 warstwa ukryta 10 neuronów", "mlpa")

# w = [np.mean(i) for i in values[:, 1]]
# o = [np.mean(i) for i in values[:, 4]]
# save_plot_2(w, o, "MLP - 1 warstwa ukryta 80 neuronów", "mlpb")

# w = [np.mean(i) for i in values[:, 2]]
# o = [np.mean(i) for i in values[:, 5]]
# save_plot_2(w, o, "MLP - 1 warstwa ukryta 400 neuronów", "mlpc")

m1 = [np.mean(i) for i in values[:, 0]]
m2 = [np.mean(i) for i in values[:, 1]]
m3 = [np.mean(i) for i in values[:, 2]]
save_plot_3(m1, m2, m3, "MLP - 1 warstwa ukryta, bez momentum", "mlp-om")

m1 = [np.mean(i) for i in values[:, 3]]
m2 = [np.mean(i) for i in values[:, 4]]
m3 = [np.mean(i) for i in values[:, 5]]
save_plot_3(m1, m2, m3, "MLP - 1 warstwa ukryta, z momentum 0.9", "mlp-wm")