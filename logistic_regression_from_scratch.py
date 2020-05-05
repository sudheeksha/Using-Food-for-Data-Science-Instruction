#This one works!

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns

def weightInitialization():
    """
    w and b are Initialized to zero
    :return:
    """
    # w = Derviative parameter 1
    # b = Derviative parameter 2/estimated later
    w = np.zeros((1,1))
    b = 0
    return w,b


def sigmoid_activation(result):
    """
    Sigmoid function.
    Function we are trying to fit
    :param result:
    :return:
    """
    final_result = 1/(1+np.exp(-result))
    return final_result


def model_optimize(w, b, X, Y):
    """
    This returns the recent cost function and new estimates using gradient descent.
    :param w: The first parameter of sigmoid function
    :param b:  The second parameter of sigmoid function
    :param X: The original data - Cooking time in secs
    :param Y: The original labeled data - (if cooked - 1, if not cooked 0)
    :return:
    """

    # given X, w, and b commute probabilty of cooked
    final_result = sigmoid_activation(np.dot(w, X.T) + b)

    # Y transpose
    Y_T = Y.T

    # length of X
    m = X.shape[0]

    cost = (-1 / m) * np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result))))
    #

    # Gradient calculation
    # How far final result is from Y value. Used to update w and b.
    dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / m) * (np.sum(final_result - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def model_predict(w, b, X, Y, learning_rate, no_iterations):
    """
    :param w: The first parameter of sigmoid function
    :param b: The second parameter of sigmoid function
    :param X: The original data - Cooking time in secs
    :param Y: The original labeled data - (if cooked - 1, if not cooked 0)
    :param learning_rate: The guessed value
    :param no_iterations:
    :return:
    """
    costs = []
    w_s = []
    b_s = []
    current_cost = 10
    previous_cost = 100
    prices = np.arange(1, 110, 1).reshape(-1, 1)
    for i in range(no_iterations):

        #
        # if current_cost > previous_cost:
        #     print("Reached break point")
        #     break
        # else:
        #     previous_cost = current_cost

        # Get the cost value and new dw and db value
        grads, cost = model_optimize(w, b, X, Y)
        current_cost = cost

        dw = grads["dw"]
        db = grads["db"]

        # weight updated
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)

        # Cost function list for plotting
        if (i % 100 == 0):
            costs.append(cost)
            w_s.append(w[0][0])
            b_s.append(b)
            # print("Cost after %i iteration is %f" %(i, cost))

    # final parameters
    coeff = {"w": w, "b": b}
    return coeff, costs,w_s, b_s


def predict(final_pred):
    y_pred = []
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def main():
    data = pd.read_csv("popcorn_v3.csv")
    data = data.head(25)
    X_cap = np.array(data['Cook time(secs)'])
    y_cap = np.array(data['edible'])
    X = X_cap.reshape(-1, 1)
    y = y_cap.reshape(-1, 1)
    n_features = X.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization()
    # Gradient Descent
    coeff, costs, w_s, b_s = model_predict(w, b, X, y, learning_rate=0.001, no_iterations=250000)
    # Final prediction
    # print(costs)
    print(w_s)
    w = coeff["w"]
    b = coeff["b"]
    print('Optimized weights', w)
    print('Optimized intercept', b)

    x = np.arange(1, 110, 1).reshape(-1, 1)
    y = sigmoid_activation(np.dot(w, x.T) + b)
    plt.plot(x,y[0])
    plt.title('Sigmoid Curve for Popcorn')
    plt.xlabel('Time in Seconds')
    plt.ylabel('Cooked?')
    plt.show()

    # print(costs)
    # print(w_s)
    # plt.plot(w_s, costs)
    # plt.show()
    # plt.plot(b_s,costs)
    # plt.show()
    # plt.plot(b_s+w_s,costs+costs)
    # plt.show()

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # #
    # fig = plt.figure()
    # ims = []
    # plt.scatter(X, y)
    # plt.xlabel('Time in secs')
    # plt.ylabel('Cooked?')
    # for i in range(len(w_s)):
    #     y = sigmoid_activation(np.dot(w_s[i], x.T) + b_s[i])
    #     line, = plt.plot(x,y[0], animated=True)
    #     ims.append([line])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    #                             repeat_delay=1000)
    # ani.save('anim2.mp4', writer=writer)

    # y_tr_pred = predict(final_train_pred, m_tr)
    # print('Training Accuracy', accuracy_score(y_tr_pred.T, y_tr_arr))
    # #
    # y_ts_pred = predict(final_test_pred, m_ts)
    # print('Test Accuracy', accuracy_score(y_ts_pred.T, y_ts_arr))

    # fig = plt.figure(figsize=(10, 6))
    # plt.xlim(0,100)
    # plt.ylim(-1.5,1.5)
    # datax = pd.DataFrame(np.column_stack([w_s, b_s]),
    #                                columns=['w_s', 'b_s'])
    # title = 'b_s'
    # plt.xlabel('weights', fontsize=20)
    # plt.ylabel('intercept', fontsize=20)
    # plt.title('Change in weight and intercepts', fontsize=20)
    #
    # def animate(i):
    #     data = datax.iloc[:int(i + 1)]  # select data range
    #     p = sns.lineplot(x=data.index, y=data[title], data=data, color="r")
    #     p.tick_params(labelsize=17)
    #     plt.setp(p.lines, linewidth=1)
    #
    # ani = matplotlib.animation.FuncAnimation(fig, animate)
    # ani.save('new1.mp4', writer=writer)

if __name__ == '__main__':
    main()