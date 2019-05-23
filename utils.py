import numpy as np

def to_categorical(value, amount):
    categorical = np.zeros(amount)
    categorical[value] = 1
    return categorical

def load_mnist_60k(amt=100000):
    X = []
    Y = []
    f = open('./mnist_train.csv', 'r')
    header = f.readline()
    counter = 0
    while True:
        line = f.readline()
        counter += 1
        if line == '' or counter > amt:
            break

        x = list(map(int, line.split(',')))
        y = to_categorical(x.pop(0), 10)

        X.append(np.array(x))
        Y.append(y)
    f.close()

    return np.array(X), np.array(Y)