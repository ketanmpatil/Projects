import numpy as np

class Perceptron():
    def __init__(self,itters=1000,learning_rt=0.001):
        
        self.weights = None # Here we are considering use of relu, (1/n_features ) if using tanh
        self.bias = 0
        self.itters = itters
        self.lr = learning_rt

    def fit(self,X,y):
        self.n_samples,self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        # Forwars
        for _ in range(self.itters):

            for idx, sample in enumerate(X):
                h = np.dot(sample,self.weights) + self.bias
                # sigmoid
                g = self._unit_step_function(h)

                cost = (1 / 2 * self.n_samples) * (y_[idx]-g) ** 2

                update = y_[idx] - g

                self.weights += self.lr * update * sample
                self.bias += self.lr * update
                
            if _ % 100 == 0:
                print(f"epoch: {_} and cost={cost}")

    def predict(self,x):
        y = np.where((np.dot(x,self.weights) + self.bias)>= 0, 1,0)
        return y

    def _unit_step_function(self,h):
        return np.where(h>=0, 1, 0)


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rt=0.01, itters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()