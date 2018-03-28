import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm


def initialize(dim1, dim2):
    """

    :param dim: size of vector w initilazied with zeros
    :return:
    """
    w = np.zeros(shape=(dim1, dim2))
    b = np.zeros(shape=(10, 1))
    return w, b


def propagate(w, b, X, Y):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :return:
    """
    m = X.shape[1]  # getting no of rows

    # Forward Prop
    A = softmax((np.dot(w.T, X) + b).T)
    cost = (-1 / m) * np.sum(Y * np.log(A))

    # backwar prop
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iters, alpha, print_cost=False):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :param num_iters: number of iterations for gradient
    :param alpha:
    :return:
    """

    costs = []
    for i in range(num_iters):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - alpha * dw
        b = b - alpha * db
        alpha = alpha * 0.99

        # Record the costs
        if i % 50 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 50 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """

    :param w:
    :param b:
    :param X:
    :return:
    """
    y_pred = np.argmax(softmax((np.dot(w.T, X) + b).T), axis=0)
    return y_pred


def model_LR(label_dict, X_train, Y_train, Y, test_x, test_y, num_iters, alpha, print_cost):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """

    w, b = initialize(X_train.shape[0], Y_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iters, alpha, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    y_prediction_train = predict(w, b, X_train)
    y_prediction_test = predict(w, b, test_x)
    print("Train accuracy: {} %", sum(y_prediction_train == Y) / (float(len(Y))) * 100)
    print("Test accuracy: {} %", sum(y_prediction_test == test_y) / (float(len(test_y))) * 100)

    d = {"costs": costs,
         "Y_prediction_test": y_prediction_test,
         "Y_prediction_train": y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": alpha,
         "num_iterations": num_iters}

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.plot()
    plt.show()
    plt.close()

    pri(X_train, y_prediction_train, label_dict)
    return d


def pri(X, Y, label):
    example = X[:, 2]
    print("Prediction for the example is ", label[Y[2]])
    plt.imshow(np.reshape(example, [28, 28]), cmap='Greys')
    plt.plot()
    plt.show()
