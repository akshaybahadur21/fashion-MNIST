import numpy as np
import pandas as pd
import fashion_LR
import fashion_nn
import fashion_DL


def main():
    label_dict = {
        0: 'T - shirt / top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankleboot'
    }

    data = pd.read_csv("fashion-mnist_train_x.csv")
    train_x = np.array(data)
    test_x = train_x[7000:10000, :]
    train_x = train_x[0:7000, :]
    data = pd.read_csv("fashion-mnist_train_y.csv")
    train_y = np.array(data)
    test_y = train_y[7000:10000, :]
    train_y = train_y[0:7000, :]
    train_y = train_y.T
    train_y = train_y[0]
    Y = train_y
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)

    choice = input("1. Logistic Regression \n2. Shallow Network \n3. Deep Network\n")
    if choice == '1':
        d = fashion_LR.model_LR(label_dict, train_x.T, train_y.T, Y, test_x.T, test_y.T[0], num_iters=1500,
                                alpha=0.000005, print_cost=True)
    elif choice == '2':
        d = fashion_nn.model_nn(label_dict, train_x.T, train_y.T, Y, test_x.T, test_y.T[0], n_h=100, num_iters=2300,
                                alpha=0.005, print_cost=True)
    elif choice == '3':
        dims = [784, 300, 100, 50, 10]
        d = fashion_DL.model_DL(label_dict, train_x.T, train_y.T, Y, test_x.T, test_y.T[0], dims, alpha=0.01,
                                num_iterations=2500, print_cost=True)
    else:
        print("Invalid Choice")


main()
