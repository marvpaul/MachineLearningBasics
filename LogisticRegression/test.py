import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class DataPointGenerator:
    '''
    Data generator which computes new random data points for 2 features
    '''
    features = None

    def __init__(self, amount_of_points) -> None:
        super().__init__()
        x1 = np.random.uniform(0, 10, amount_of_points)
        x2 = np.random.uniform(1, 50, amount_of_points)
        self.features = np.concatenate([x1.reshape((-1, 1)), x2.reshape((-1, 1))],
                                       axis=1)


generator = DataPointGenerator(100)
X = generator.features

#Feature scaling
def scalingFeature(feature):
    '''
    Scaling a given features to -1 to 1
    :param feature: given features
    :return: scaled features
    '''
    u = np.sum(feature) / feature.size
    std = np.sqrt(np.abs(np.square(u) - np.square(feature)))
    return (feature - u) / std
X[:, 0] = scalingFeature(X[:, 0])
X[:, 1] = scalingFeature(X[:, 1])
y = np.zeros(100)
y[:50,] = 1
# Anpassen der Datenstruktur an die Anforderungen des Algorithmus
TRAIN = np.transpose(X)

def feature_scaling(feature):
    return (feature - feature.mean()) / feature.std()

for key, feature in enumerate(TRAIN):
    TRAIN[key] = feature_scaling(TRAIN[key])

def sigmoid(x):
    return lambda x: 1/(1 + np.exp(-x))

x = np.linspace(-5, 5, 50)
f = sigmoid(x)

# Visualisierung
plt.xlim(-5, 5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, f(x))


thetas = np.array([1.1, 2.0, -.9])

def logistic_hypothesis(thetas):
    t = thetas[1:]
    return lambda x: 1/(1 + np.exp(np.dot(-x.transpose(),t)))

h = logistic_hypothesis(thetas)

print(h(TRAIN))

def cross_entropy_loss(h, X, Y):
    return lambda x: -Y*np.log(h(X)) - (1 - Y)*np.log(1 - h(X))

loss = cross_entropy_loss(h, TRAIN, y)

l = loss(thetas)

def cost_function(X, Y):
    m = X.shape[1]
    def f(thetas):
        h = logistic_hypothesis(thetas)
        return (-1./m) * (Y*np.log(h(X)) + (1 - Y)*np.log(1 - h(X))).sum()

    return f

j = cost_function(TRAIN, y)
print(j(thetas))

def compute_new_theta(x, y, thetas, alpha, h):
    m = len(y)
    return thetas - (alpha / m) * (((h - y) * x)).sum()


#alpha = np.linspace(0.01, 0.001, 5)
alpha = np.array([0.01])
iterations = 20000
thetas = np.array([1, 3, 5])
def gradient_descent(alpha, thetas, iterations, x, y):
    costs = np.zeros([len(alpha), iterations])
    for a in range(len(alpha)):
        thetas_sum = np.zeros([iterations])
        for i in range(0, iterations):
            h = logistic_hypothesis(thetas)
            thetas = compute_new_theta(x, y, thetas, alpha[a], h(TRAIN))
            costs[a][i] = j(thetas)
            thetas_sum[i] = thetas.sum()
    return thetas_sum, costs, thetas

thetas_sum, costs, thetas_opt = gradient_descent(alpha,thetas,iterations, TRAIN, y)

print(thetas_opt)

fig = plt.figure()
plt.xlabel('Iterationen')
plt.ylabel('Kosten')

for i, c in enumerate(costs):
    plt.plot(costs[i])
plt.show()