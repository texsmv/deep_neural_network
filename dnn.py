import numpy as np

def sigmoidea(x):
    return 1 / (1 + np.e ** -x)

class dnn:
    def __init__(self):

        self.layers = []
        self.weights = []
        self.bias = []


    def add_layer(self, size):
        self.layers = self.layers + [np.empty([size, ])]
        self.bias = self.bias + [np.ones([size, ])]
        self.weights = self.weights + [np.ones([size, len(self.layers[len(self.layers) - 2])])]

    def add_input_layer(self, size):
        self.layers = self.layers + [np.empty([size, ])]

    def imprimir(self):
        print("Input layer: ")
        print(self.layers[0])
        for i in range(1, len(self.layers)):
            print("Layer: ")
            print(self.layers[i])
            print("Bias: ")
            print(self.bias[i - 1])
            print("Pesos: ")
            print(self.weights[i - 1])
            print(" ")

    def calcular_netas(self):
        for i in range(1, len(self.layers)):
            self.layers[i] = np.matmul(self.weights[i - 1], self.layers[i - 1])
            for j in range(0, len(self.layers[i])):
                self.layers[i][j] = sigmoidea(self.layers[i][j]) + self.bias[i - 1][j]

    def forward(self, input):
        self.layers[0] = input
        self.calcular_netas()

rnn = dnn()
rnn.add_input_layer(3)
rnn.add_layer(3)
rnn.add_layer(2)
rnn.add_layer(5)
rnn.forward([1,1,1])
rnn.imprimir()
