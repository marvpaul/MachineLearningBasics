import numpy as np
import matplotlib.pyplot as plt

'''
Generate some data 
'''
# class 0:
# covariance matrix and mean

cov0 = np.array([[5, -3], [-3, 3]])

mean0 = np.array([2., 3])

# number of data points
m0 = 1000
# generate m0 gaussian distributed data points with
# mean0 and cov0.
r0 = np.random.multivariate_normal(mean0, cov0, m0)

# covariance matrix
cov1 = np.array([[5, -3], [-3, 3]])
mean1 = np.array([1., 1])
m1 = 1000
r1 = np.random.multivariate_normal(mean1, cov1, m1)

x = np.concatenate((r0, r1))
y = np.zeros(len(r0) + len(r1))
y[:len(r0), ] = 1


# 1) Erstellen Sie eine Pythonfunktion die, die
# logistische Funktion berechnet.
# Stellen Sie diese im Bereich [-5, 5] graphisch dar.
def logistic_function():
    def h(x):
        return 1 / (1 + np.exp(-x))

    return h


# -----------OKAY ----------------

x_log = np.arange(-5, 5, 0.1)
y_log = logistic_function()(x_log)

plt.plot(x_log, y_log)
plt.show()


# -----------OKAY ----------------


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


theta = np.array([1.1, 2.0, -0.9])
h = logistic_hypothesis(theta)
print("Logistic function for: [1, 2]:", h(np.array([[1, 2]])))


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
        return -y * np.log(h(X)) - (1 - y) * np.log(1 - h(X))

    return loss


def squared_error_loss(h, X, y):
    return 1 / 2 * np.square(h(X) - y)


loss = cross_entropy_loss(x, y)


# 4) Implementieren Sie die Kostenfunktion J als Python Funktion:
# cost_function(X, y, h, loss)
# zusätzlich zu X und y soll die Funktion die Hypothese h
# und den Loss aufnehmen.
#
# Die Pythonfunktion soll dabei eine Funktion zurückgeben, die
# den Parametervektor theta aufnimmt.

def cost_function(X, y, h, loss):
    def costs(theta):
        loss_computed = loss(theta)
        return 1. / len(X) * np.sum(loss_computed)

    return costs


def cost_function_lam_reg(X, y, h, loss):
    def costs(theta, alpha):
        loss_computed = loss(theta)
        return 1. / len(X) * np.sum(loss_computed) + alpha * np.sum(np.square(theta))

    return costs


print("Costs for good theta", cost_function(x, y, h, loss)(theta))

print("Costs for bad theta", cost_function(x, y, h, loss)(np.array([1.4, 0.44, -0.55])))


# 5) Implementieren Sie das Gradientenabstiegsverfahren unter Benutzung der Kostenfunktion und der Hypothese.
# 5a) Schreiben Sie eine Funktion die die Update Rules anwendet zur Berechnung der neuen theta-Werte:
# theta = compute_new_theta(x, y, theta, alpha, hypothesis)
# 5b) Wählen Sie Startwerte in der Umgebung des Miniums der Kostenfunktion für theta.
# Wenden Sie iterativ die compute_new_theta Funktion an und finden Sie so ein Theta mit niedrigen Kosten.
# Kapseln Sie dies in eine Funktion:
# theta = gradient_descent(alpha, theta, nb_iterations, X, y)

def compute_new_theta(x, y, theta, alpha):
    '''
    Compute new theta values for multivariate linear regression with gradient descent
    :param x: features
    :param y: y values for given feature values
    :param theta: array with theta values
    :param alpha: learning rate
    :return:
    '''
    hypothesis = logistic_hypothesis(theta)
    m = len(y)
    x_temp = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    change = hypothesis(x) - y
    return theta - ((alpha / m) * np.dot(change, x_temp))


def compute_new_theta_lam_reg(x, y, theta, alpha, alpha_reg):
    '''
    Compute new theta values for multivariate linear regression with gradient descent
    :param x: features
    :param y: y values for given feature values
    :param theta: array with theta values
    :param alpha: learning rate
    :return:
    '''
    hypothesis = logistic_hypothesis(theta)
    m = len(y)
    x_temp = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    change = hypothesis(x) - y
    return theta - ((alpha / m) * np.dot(change, x_temp + 2 * alpha_reg / m * theta))


def gradient_descent(alpha, theta_, nb_iterations, x, y):
    '''
    Gradient descent for multivariate linear regression
    :param alpha: learning rate
    :param theta_: array with theta values
    :param nb_iterations: number of iterations the gradient descent should do
    :param x: features
    :param y: y values
    :return: new computed theta values
    '''
    n_theta = theta_
    costs = []
    for i in range(nb_iterations):
        n_theta = compute_new_theta(x, y, n_theta, alpha)

        costs.append(cost_function(x, y, logistic_hypothesis(n_theta), cross_entropy_loss(x, y))(n_theta))
    print(costs)
    plt.plot(np.arange(0, nb_iterations), costs)
    plt.show()
    return n_theta


def gradient_descent_lam_reg(alpha_learningrate, theta_, nb_iterations, x, y, alpha_lam_reg):
    '''
    Gradient descent for multivariate linear regression
    :param alpha: learning rate
    :param theta_: array with theta values
    :param nb_iterations: number of iterations the gradient descent should do
    :param x: features
    :param y: y values
    :return: new computed theta values
    '''
    n_theta = theta_
    costs = []
    for i in range(nb_iterations):
        n_theta = compute_new_theta_lam_reg(x, y, n_theta, alpha_learningrate, alpha_lam_reg)

        costs.append(cost_function(x, y, logistic_hypothesis(n_theta), cross_entropy_loss(x, y))(n_theta))
    print(costs)
    plt.plot(np.arange(0, nb_iterations), costs)
    plt.show()
    return n_theta


new_theta = gradient_descent_lam_reg(0.1, np.array([-1, -2, 2]), 1000, x, y, 0.01)
print(new_theta)


# 6) Zeichen Sie die Entscheidungsebene in den Scatter-Plot der Daten
# Hinweis: Für diese gilt: theta[0] + theta[1] * x1 + theta[2] * x2 = 0.5

def decision_boundary(theta):
    x1_range = np.arange(-10, 10, 0.01)
    x2_range = np.arange(-10, 10, 0.01)
    x1_range, x2_range = np.meshgrid(x1_range, x2_range)
    plt.scatter(r0[..., 0], r0[..., 1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[..., 0], r1[..., 1], c='r', marker='x', label="Klasse 1")
    plt.xlabel("x0")
    plt.ylabel("x1")

    z = np.around(logistic_function()(theta[0] + theta[1] * x1_range + theta[2] * x2_range))
    plt.contour(x1_range, x2_range, z)
    plt.show()


decision_boundary(new_theta)

# 7) Berechnen Sie den Klassifikationsfehler, d.h. der Anteil
# der falsch-klassifizierten Datensätze:
# Klassifikationsfehler = Anzahl der falsch-klassifizierten Datensätze / Anzahl der Datensätze

h_pred = logistic_hypothesis(new_theta)
y_pred = h_pred(x)
y_pred = np.around(y_pred)
false_classifications = 0
for i in range(y.size):
    if (y_pred[i] != y[i]):
        false_classifications += 1

print("False classifications: ", false_classifications)
