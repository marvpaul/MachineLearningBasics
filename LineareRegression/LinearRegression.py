import DataPointGenerator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

generator = DataPointGenerator.DataPointGenerator(100)
x = generator.points[:, 0]
y = generator.points[:, 1]


# 2)
#   Implementieren Sie die Hypothese (lineares Modell) als Python Funktion:
#       linear_hypothesis(theta_0, theta_1)
#   Die Pythonfunktion soll dabei eine Funktion zurückgeben:
#   >> hypothesis = linear_hypothesis(2., 3.)
#   >> print hypothesis(np.array([1., 2.]))
#   [ 5.  8.]
def linear_hypothesis(theta_0, theta_1):
    def fct(x):
        return theta_0 + theta_1 * x

    return fct


hypothesis = linear_hypothesis(2., 3.)
print(hypothesis(np.array([1., 2.])))


# 3)
#    Implementieren Sie die Kostenfunktion J als Python Funktion:
#       def cost_function(hypothesis, x, y):

#              TODO

#
#    Die Pythonfunktion soll dabei eine Funktion zurückgeben, die
#    die beiden Parameter theta_0 und theta_1 aufnimmt.
#
#   >> j = cost_function(linear_hypothesis, x, y)
#   >> print j(2.1, 2.9)
#
def cost_function(hypothesis, x, y):
    def fct(theta1, theta2):
        curr_hypothesis = hypothesis(theta1, theta2)
        return 1 / (2 * x.size) * np.sum(np.square(curr_hypothesis(x) - y))

    return fct


j = cost_function(linear_hypothesis, x, y)
print(j(10, 5))


# 4)
#   Plotten Sie die Kostenfunktion in der Umgebung des Minimums als Contourplot.
#   Verwenden Sie hierzu plt.contour(X,Y,Z) und zum Erzeugen des X-Y-Oberflaechengitters meshgrid(..)
ran = 4
a = 2
b = 3
t0 = np.arange(a - ran, a + ran, ran * 0.05)
t1 = np.arange(b - ran, b + ran, ran * 0.05)

C = np.zeros([len(t0),len(t1)])
c = cost_function(linear_hypothesis, x, y)

for i, t_0 in enumerate(t0):
    for j, t_1 in enumerate(t1):
        C[j][i] = c(t_0, t_1)

T0, T1 = np.meshgrid(t0, t1)

plt.subplot(121)
plt.contour(T0, T1, C)
plt.xlabel('$\Theta_0$')
plt.ylabel('$\Theta_1$')
plt.title('Kostenfunktion')
plt.show()


# 5)
#   Implementieren Sie das Gradientenabstiegsverfahren unter Benutzung der Kostenfunktion und der linearen Hypothese.
#  5a) Schreiben Sie eine Funktion die die Update Rules anwendet zur Berechnung der neuen theta-Werte:
#      theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, alpha)
#
#  5b) Wählen Sie Startwerte in der Umgebung des Miniums der Kostenfunktion für theta.
#      Wenden Sie iterativ die compute_new_theta Funktion an und finden Sie so ein Theta mit niedrigen Kosten.
#
#  5c) Plotten Sie den Fortschritt (Verringerung der Kosten über den Iterationen) für 5b
def compute_new_theta(x, y, theta_0, theta_1, alpha):
    hypothesis = linear_hypothesis(theta_0, theta_1)(x)
    theta_0_temp = theta_0 - alpha * (1 / x.size) * (np.sum(hypothesis - y))
    theta_1_temp = theta_1 - alpha * (1 / x.size) * (np.sum((hypothesis - y) * x))
    return theta_0_temp, theta_1_temp

def train(initial_theta_0, initial_theta_1, alpha, iterations):
    theta_0 = initial_theta_0
    theta_1 = initial_theta_1
    j = cost_function(linear_hypothesis, x, y)

    costs = []
    for i in range(iterations):
        theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, alpha)
        costs.append(j(theta_0, theta_1))
    plt.plot(np.arange(0, iterations, 1), costs)
    plt.show()

train(8, 7, 0.001, 100)
'''
theta_0 = 100
theta_1 = 5
alpha = 0.0005
ITERATIONS = 100000
costs = []
for i in range(ITERATIONS):
    theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, alpha)
    costs.append(j(theta_0, theta_1))
#plt.plot(np.arange(0, ITERATIONS, 1), costs)
#plt.show()

# 6)
#   Plotten Sie das Modell (Fit-Gerade) zusammen mit den Daten.
plt.scatter(x, y, label='datapoints', color="red")
x = np.arange(-50, 150, 1)
y = theta_0 + theta_1 * x
plt.plot(x, y, label='$f(x) = {theta0}x + {theta1}$'.format(theta0=theta_0, theta1 = theta_1))
plt.legend(loc='best')
plt.show()

# 7)
#    Trainieren (siehe 5b) für verschiedene Werte der Lernrate und
#    plotten Sie Kosten über den Iterationen in einen Graph.

#TODO
'''