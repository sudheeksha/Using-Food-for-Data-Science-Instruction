import numpy as np # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
from sklearn.preprocessing import MinMaxScaler # for normalization
from sklearn.utils import shuffle
from matplotlib import pyplot as plt # For plotting graphs
import seaborn as sns; sns.set(font_scale=1.2) # For plotting graphs

# Global variables
regularization_strength = 10000
learning_rate = 0.00001


def compute_cost(W, X, Y):
    """
    Cost function --> https://miro.medium.com/max/1084/1*6w_B_DjhGvaqCnvhzhhkDg.png
    This function is used to calculate the cost function/objective function.
    Remember: Trying to reduce the cost function
    :param W: weights
    :param X: features
    :param Y: target
    :return: cost
    """

    # length of X
    N = X.shape[0]

    # calculate max(0, 1 - y_i * (w.x + b)
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


def calculate_gradient(W, X_batch, Y_batch):
    """
    Gradient cost function --> https://miro.medium.com/max/1386/1*ww3F21VMVGp2NKhm0VTesA.png
    :param W: The weights
    :param X_batch: feature
    :param Y_batch: target
    :return: dw
    """
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    # calculate max(0, 1 - y_i * (w.x + b)
    distance = 1 - (Y_batch * np.dot(X_batch, W))

    dw = np.zeros(len(W))

    # Check if distance = 0 if yes, dw = W else, dw = w - C y_i x_i
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    # dw/N
    dw = dw/len(Y_batch)  # average

    return dw


def svm(features, outputs):
    """
    SVM using gradient descent
    :param features: The features Sugar and Flour
    :param outputs: Tells if its a cupcake or muffin in 1 or -1
    :return: The weights calculated
    """

    # Max number of iterations
    max_epochs = 100000

    # Initialize the weights to 0
    weights = np.zeros(features.shape[1])
    nth = 0

    # Initialize previous cost to 0
    prev_cost = float("inf")

    cost_threshold = 0.01  # in percent

    # gradient descent begins
    for epoch in range(1, max_epochs):

        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)

        for ind, x in enumerate(X):

            gradient = calculate_gradient(weights, x, Y[ind])

            # calculate the new weight
            weights = weights - (learning_rate * gradient)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:

            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))

            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1

    return weights


def main():
    """
    This is the main function.
    :return:
    """
    # Reading the data.
    data = pd.read_csv('/Users/sudheekshagarg/PycharmProjects/capstone/svm/recipes_muffins_cupcakes.csv')

    # SVM only accepts numerical values.
    # Therefore, we will transform the categories Muffins and Cupcakes into
    # values 1 and -1 (or -1 and 1), respectively.
    recipe_map = {'Muffin':1.0, 'Cupcake':-1.0}
    data['Type'] = data['Type'].map(recipe_map)

    # Defining Y and X variable.
    # Y is the target
    Y = data.loc[:, 'Type']  # all rows of 'Type'

    # X is features.
    x = data[['Flour', 'Sugar']]  # all rows of column 1 and ahead (features)

    # Normalize the features using MinMaxScalar
    X_normalized = MinMaxScaler().fit_transform(x.values)
    X = pd.DataFrame(X_normalized)

    print("training started...")

    # The weights returned from svm function
    W = svm(X.to_numpy(), Y.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # Plotting the graph
    sns.lmplot('Flour', 'Sugar', data=data, hue='Type', fit_reg=False)

    x2 = [W[0], W[1], -W[1], W[0]]
    x3 = [W[0], W[1], W[1], -W[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)

    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')

    plt.show()

if __name__ == '__main__':
    main()

