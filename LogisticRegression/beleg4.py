import numpy as np
import matplotlib.pyplot as plt

def target_fun(x):
    return np.sin(x)


def linear_hypothesis_h1(theta_0, theta_1):
    def fct(x):
        return theta_0 + theta_1 * x

    return fct

def linear_hypothesis_h0(theta_0):
    def fct(x):
        return np.ones(shape=x.shape) * theta_0

    return fct

def get_loss(hypothesis, samples):
    x = samples.transpose()[0,0:]
    y = samples.transpose()[1,0:]
    return 1/samples.shape[0] * np.sum(np.square(hypothesis(x) - y))

#Hypothesenmenge H_0: h = theta_0
def get_h0(samples):
   return linear_hypothesis_h0(samples[0][1] + samples[1][1] / 2)

#Hypothesenmenge H_1: h = theta_0 + theta_1 * x
def get_h1(samples):
    '''
    Get h1 hypthesis with 2 samples given
    :param samples: two samples in multidimensional array, e.g. [[-2, -4], [0, 1]]
    :return: linear hypothesis with calculated theta vlaues
    '''
    theta1 = (samples[1][1]-samples[0][1]) / (samples[1][0] - samples[0][0])
    theta0 = samples[0][1] - (theta1 * samples[0][0])
    return linear_hypothesis_h1(theta0, theta1), (theta0, theta1)

#Erzeugen Sie z.B. 10000 verschiedene Trainingsdaten mit jeweils zwei Beispielen
rand_values = np.random.rand(2*10000)*2*np.pi
fun_values = target_fun(rand_values)
two_samples_for_each = np.array((rand_values, fun_values)).transpose().reshape((10000, 2, 2))
print(get_h1([[-2, -4], [0, 1]])[0](-1))
print(get_h0([[-2, 0], [0, 2]])(np.array(10)))

#Get some training data
rand_values = np.random.rand(1000*10000)*2*np.pi
fun_values = target_fun(rand_values)
twenty_train_for_each = np.array((rand_values, fun_values)).transpose().reshape((10000, 1000, 2))

h1 = get_h1([[-2, -4], [0, 1]])

#Get the hypothesis
hypothesis_h1 = []
hypothesis_h0 = []
for two_samples in two_samples_for_each:
    hypothesis_h1.append(get_h1(two_samples))
    hypothesis_h0.append(get_h0(two_samples))

#This is a test loss, it should be zero because the data are exaclty fitting the hypothesis
sample_vals = np.array([[-1, h1[0](-1)], [-2, h1[0](-2)]])
print(get_loss(h1[0], sample_vals))

#Another loss which shouldn't be 0
print(get_loss(h1[0], twenty_train_for_each[1]))

e_out_h0 = np.zeros(10000)
e_out_h1 = np.zeros(10000)
for i in range(len(hypothesis_h0)):
    e_out_h1[i] = (get_loss(hypothesis_h1[i][0], twenty_train_for_each[i]))
    e_out_h0[i] = (get_loss(hypothesis_h0[i], twenty_train_for_each[i]))

def plot_hypothesis(hypothesis_h0, hypothesis_h1):
    x = np.arange(0, 2*np.pi, 0.1)
    y_0 = hypothesis_h0(x)
    y_1 = hypothesis_h1(x)
    sin = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y_0)
    ax.plot(x, y_1)
    ax.plot(x, sin)
    plt.show()


average_h0_hypthesis = 0

for hypothessis in hypothesis_h0:
    average_h0_hypthesis += hypothessis(np.array(100))

average_h0_hypthesis = average_h0_hypthesis/len(hypothesis_h0)
print("========Hypothesis 0==========")
print("theta0 -->", average_h0_hypthesis)

average_theta0 = 0
average_theta1 = 0
for hypothessis in hypothesis_h1:
    average_theta0 += hypothessis[1][0]
    average_theta1 += hypothessis[1][1]

average_theta0 = average_theta0/len(hypothesis_h0)
average_theta1 = average_theta1/len(hypothesis_h0)
print("========Hypothesis 1==========")
print("theta0 -->", average_theta0)
print("theta1 -->", average_theta1)

plot_hypothesis(linear_hypothesis_h0(average_h0_hypthesis), linear_hypothesis_h1(average_theta0, average_theta1))
e_out_h0_average = get_loss(linear_hypothesis_h0(average_h0_hypthesis), twenty_train_for_each[1])

x_vals = np.arange(0, 2*np.pi, 0.1)
y_vals_h0 = linear_hypothesis_h0(average_h0_hypthesis)(x_vals)
y_vals_sin = target_fun(x_vals)

bias_h1 = np.mean(np.square(linear_hypothesis_h1(average_theta0, average_theta1)(x_vals) - y_vals_sin))
bias_h0 = np.mean(np.square(linear_hypothesis_h0(average_h0_hypthesis)(x_vals) - y_vals_sin))

print(bias_h0)
print(bias_h1)
means_hd0 = np.empty(shape=(len(hypothesis_h0), 1))
means_hd1 = np.empty(shape=(len(hypothesis_h0), 1))
for i in range(len(hypothesis_h0)):
    means_hd1[i] = np.mean(np.square(hypothesis_h1[i][0](x_vals) - linear_hypothesis_h1(average_theta0, average_theta1)(x_vals)))
    means_hd0[i] = np.mean(np.square(hypothesis_h0[i](x_vals) - linear_hypothesis_h0(average_h0_hypthesis)(x_vals)))

var_h0 = np.mean(means_hd0)
var_h1 = np.mean(means_hd1)
print(var_h0)
print(var_h1)



'''
e_out_h0_new = 0
for hypothessis_ in hypothesis_h0:
    e_out_h0_new += 1/len(hypothesis_h0) * get_loss(hypothessis_, twenty_train_for_each[0])
print(e_out_h0_new)
var_h0 = 0
for hypothessis in hypothesis_h0:
    var_h0 += 1/len(x_vals) * np.sum(np.square(hypothessis(x_vals) - linear_hypothesis_h0(average_h0_hypthesis)(x_vals)))
var_h0 /= len(hypothesis_h0)
bias_h0 = np.sum(np.square(linear_hypothesis_h0(average_h0_hypthesis)(x_vals), y_vals_sin))

print("========Hypothesis 0==========")
print(e_out_h0_average)
print("bias:", bias_h0, ", var:", var_h0)

e_out_h1_average = get_loss(linear_hypothesis_h1(average_theta0, average_theta1), twenty_train_for_each[1])
y_vals_h1 = linear_hypothesis_h1(average_theta0, average_theta1)(x_vals)
bias_h1 = np.sum(np.abs(y_vals_h1 - y_vals_sin)) / x_vals.shape[0]
var_h1 = e_out_h1_average - bias_h1 * bias_h1
print("========Hypothesis 1==========")
print("bias:", bias_h1, ", var:", var_h1)
print(e_out_h1_average)

'''