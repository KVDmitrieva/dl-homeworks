import numpy as np
from scipy.special import logsumexp, softmax
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = np.where(input > 0, input, 0)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (input > 0).astype(float)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigmoid = 1 / (1 + np.exp(-input))
        return grad_output * (sigmoid - sigmoid ** 2)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = softmax(input, axis=1)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        m, n = input.shape
        sftmx = softmax(input, axis=1)

        tensor1 = np.einsum('ij,ik->ijk', sftmx, sftmx)
        tensor2 = np.einsum('ij,jk->ijk', sftmx, np.eye(n, n))

        return np.einsum('ijk, ik->ij', tensor2 - tensor1, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        m, n = input.shape
        self.output = input - logsumexp(input, axis=1).reshape((m, 1))
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        m, n = input.shape
        sftmx = softmax(input, axis=1)

        tensor1 = np.einsum('ij,ik->ijk', sftmx, np.ones((m, n)))
        tensor2 = np.einsum('ij,jk->ijk', np.ones((m, n)), np.eye(n, n))

        return np.einsum('ijk, ik->ij', tensor2 - tensor1, grad_output)
