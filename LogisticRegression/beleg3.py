import numpy as np
import matplotlib.pyplot as plt


class DataPointGenerator:
    '''
    Data generator which computes new random data points for 2 features
    '''
    features = None

    def __init__(self, amount_of_points) -> None:
        super().__init__()
        x1 = np.random.uniform(3, 10, amount_of_points)
        x2 = np.random.uniform(3, 5, amount_of_points)
        self.features =  np.concatenate([x1.reshape((-1,1)), x2.reshape((-1, 1))],
                                        axis=1)


generator = DataPointGenerator(100)
x = generator.features


# 1) Erstellen Sie eine Pythonfunktion die, die
# logistische Funktion berechnet.
# Stellen Sie diese im Bereich [-5, 5] graphisch dar.
def logistic_function():
    def h(x):
        return 1 / (1 + np.exp(-x))

    return h


x_log = np.arange(-5, 5, 0.1)
y_log = logistic_function()(x_log)

plt.plot(x_log, y_log)
plt.show()

# 2) Implementieren Sie die Hypothese als Python Funktion:
# logistic_hypothesis(theta)
# Die Pythonfunktion soll dabei eine Funktion zurückgeben:
# >> theta = np.array([1.1, 2.0, -.9])
# >> h = logistic_hypothesis(theta)
# >> print h(X)
# array([ -0.89896965, 0.71147926, ....

def logistic_hypothesis(theta):
    def h(x):
        # Add row with ones
        x_temp = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

        return logistic_function()(x_temp.dot(theta))

    return h


theta = np.array([1.1, 2.0, -.9])
h = logistic_hypothesis(theta)
print(h(np.array([[1, 2], [1, 2]])))

# 3) Implementieren Sie den Cross-Entropy-Loss und
# den Squared-Error-Loss als Python Funktion.
# Die Pythonfunktion soll dabei eine Funktion zurückgeben:

# >> loss = cross_entropy_loss(h, X, y)
# >> print loss(theta)
# array([ 7.3, 9.5, ....
# Rückgabevektor hat m-Elemente (Anzahl der Datensätze)

def cross_entropy_loss(X, y):
    def loss(theta):
        h = logistic_hypothesis(theta)
        pos_values = y * np.log(h(X))
        neg_values = np.log(1-h(X))
        sum_cross_entr = np.sum(pos_values + neg_values)
        return -1 / len(X) * sum_cross_entr
    return loss

def squared_error_loss(h, X, y):
    return 1 / (2*len(X)) * np.sum(np.square(h(X)-y))

print(cross_entropy_loss(x, h(x))(np.array([100, 10, -100])))

print(cross_entropy_loss(x, h(x))(theta))
