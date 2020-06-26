from NeuralNet import NeuralNet
import math
import random
import csv

def main():
    random.seed(1)
    results_sigmoid_iris = 0.0
    results_tanh_iris = 0.0
    results_sigmoid_seed = 0.0
    results_tanh_seed = 0.0
    test_runs = 20

    iris_data = import_data('IRIS.csv')
    seed_data = import_data('SEEDS.csv')

    for test_run in range(test_runs):
        print("Testing sigmoid function on iris data...")
        results_sigmoid_iris += cross_validate(iris_data)
        print()
        print("Testing tanh function on iris data...")
        results_tanh_iris += cross_validate(iris_data, activate=(tanh, tanh_derivative))
        print()

    for test_run in range(test_runs):
        print("Testing sigmoid function on seed data...")
        results_sigmoid_seed += cross_validate(seed_data, hidden_layers=12)
        print()
        print("Testing tanh function on seed data...")
        results_tanh_seed += cross_validate(seed_data, hidden_layers=12, activate=(tanh, tanh_derivative))
        print()

    print("sigmoid accuracy on iris data: ", results_sigmoid_iris / test_runs)
    print("tanh accuracy on iris data: ", results_tanh_iris / test_runs)
    print("sigmoid accuracy on seeds data: ", results_sigmoid_seed / test_runs)
    print("tanh accuracy on seeds data: ", results_tanh_seed / test_runs)

def import_data(file_name):
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        data = [line for line in reader]
    return data

def cross_validate(data, hidden_layers=7, test_percentage=0.2, **kwargs):
    random.shuffle(data)
    training_samples = list(data)
    testing_samples = []

    for sample in range((int) (len(data) * test_percentage)):
        testing_samples.append(training_samples.pop(1))

    training_data = random.choices(training_samples,k=1000000)
    network = NeuralNet([len(training_data[0]) - 1, hidden_layers, num_outputs(data)])

    if('activate' in kwargs):
        network.activate = kwargs['activate']

    network.train(training_data)
    return network.test(testing_samples)

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def tanh(z):
    return 2.0 * sigmoid(2 * z) - 1.0

def tanh_derivative(z):
    return 1.0 - tanh(z) ** 2

def num_outputs(data):
    return len({data[outpt][-1] for outpt in range(len(data))})

if __name__ == "__main__": main()
