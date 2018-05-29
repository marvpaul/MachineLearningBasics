import numpy as np


def target_fun(x):
    return np.sin(x)


def linear_hypothesis_h1(theta_0, theta_1):
    def fct(x):
        return theta_0 + theta_1 * x

    return fct

def linear_hypothesis_h0(theta_0):
    def fct(x):
        return theta_0

    return fct

def get_h0(samples):
   return linear_hypothesis_h0(samples[0][1] + samples[1][1] / 2)

def get_h1(samples):
    '''
    Get h1 hypthesis with 2 samples given
    :param samples: two samples in multidimensional array, e.g. [[-2, -4], [0, 1]]
    :return: linear hypothesis with calculated theta vlaues
    '''
    theta1 = (samples[1][1]-samples[0][1]) / (samples[1][0] - samples[0][0])
    #y - (m*x)
    theta0 = samples[0][1] - (theta1 * samples[0][0])
    return linear_hypothesis_h1(theta0, theta1)

rand_values = np.random.rand(2*10000)*2*np.pi
fun_values = target_fun(rand_values)
two_samples_for_each = np.array((rand_values, fun_values)).transpose().reshape((10000, 2, 2))
print("Sth")
print(get_h1([[-2, -4], [0, 1]])(-1))
print(get_h0([[-2, 0], [0, 2]])(10))

