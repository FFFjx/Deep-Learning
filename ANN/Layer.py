import numpy as np
from functools import reduce


# -----------------------------------------------------------------------------------------------------Convolution layer
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        :param in_channels: input channel
        :param out_channels: output channel
        :param kernel_size: the size of convolution kernel
        :param stride: stride
        :param padding: padding length for each side of an image
        :param bias: whether using bias
        """
        self.channel_in = in_channels
        self.channel_out = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        filter_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = np.random.randn(*filter_shape) * (2 / reduce(lambda x, y: x * y, filter_shape[1:])) ** 0.5
        self.need_bias = bias
        if self.need_bias:
            self.bias = np.random.randn(self.channel_out)
        else:
            self.bias = None

    def forward(self, input):
        """
        :param input: feature map. dim：[N, C, H, W]
        :return: feature map. dim:[N, O, H, W]
        """
        self.input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                            'constant')

        # # Directly do the convolution operation without acceleration.
        # out_h = int((input.shape[2] - self.kernel + 2 * self.padding) / self.stride + 1)
        # out_w = int((input.shape[3] - self.kernel + 2 * self.padding) / self.stride + 1)
        #
        # output = np.zeros((self.input.shape[0], self.channel_out, out_h, out_w))  # [N, O, H', W']
        # for i in range(output.shape[0]):
        #     for j in range(output.shape[1]):  # j indicates the number of conv kernel
        #         output[i:i + 1, j:j + 1, :, :] += self.bias[j]
        #         for k in range(output.shape[2]):
        #             for l in range(output.shape[3]):
        #                 receptive_field = self.input[i:i + 1, :,
        #                                   (self.stride * k):(self.weight.shape[2] + self.stride * k),
        #                                   (self.stride * l): (self.weight.shape[3] + self.stride * l)]
        #                 kernel = self.weight[j:j + 1, :, :, :]
        #                 result = np.multiply(kernel, receptive_field)
        #                 output[i][j][k][l] += np.sum(result)

        # Transform convolution into matrix multiplication in order to speed up operation. In the other hand, it
        # increases storage occupation.
        N, C, H, W = self.input.shape
        O, _, K, _ = self.weight.shape
        out_h = int((H - K) / self.stride + 1)
        out_w = int((W - K) / self.stride + 1)
        x_cols = self.img2col(self.input, self.weight, self.stride)
        weight_cols = self.weight.reshape(O, -1).T
        output = np.dot(x_cols, weight_cols)
        if self.need_bias:
            output += self.bias
        output = output.reshape((N, output.shape[0] // N, -1)).reshape((N, out_h, out_w, O))
        output = np.transpose(output, (0, 3, 1, 2))
        return output

    def backward(self, eta, learning_rate):
        """
        :param eta: gradient that the next layer return
        :param learning_rate: learning rate
        :return: gradient of this layer
        """
        x = self.input.transpose((1, 0, 2, 3))
        delta_kernel = eta.transpose((1, 0, 2, 3))
        self.b_grad = eta.sum(axis=(0, 2, 3))
        self.w_grad = self.conv(x, delta_kernel, self.stride, 0)
        self.w_grad = np.swapaxes(self.w_grad, 0, 1)
        self.weight -= learning_rate * self.w_grad / eta.shape[0]
        if self.need_bias:
            self.bias -= learning_rate * self.b_grad / eta.shape[0]

        # backprop
        padding = int(((self.input.shape[2] - 1) * self.stride + self.weight.shape[2] - eta.shape[2]) / 2)
        delta_new = np.pad(eta, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
        weight_flip = np.flip(self.weight, (2, 3))  # convolution kernel flip 180°
        weight_flip = np.swapaxes(weight_flip, 0, 1)  # swap the dimension of channel_in and channel_out[C,O,K,K]
        delta_new = self.conv(delta_new, weight_flip, self.stride, 0)
        return delta_new

    # def img2col(self, img, kernel, bias, stride):
    #     """
    #     :param img: image input had been padding. dim:[N, C, H, W]
    #     :param kernel: convolution kernel. dim:[O, C, K, K]
    #     :param bias: bias
    #     :param stride: stride
    #     :return: convolution result
    #     """
    #     N, C, H, W = img.shape
    #     O, _, K, _ = kernel.shape
    #     out_h = int((H - K) / stride + 1)
    #     out_w = int((W - K) / stride + 1)
    #     result = np.zeros((N, O, out_h, out_w))
    #
    #     kernel_flatten = np.zeros_like((kernel.reshape(-1, kernel.shape[0])))
    #     for i in range(kernel.shape[0]):
    #         kernel_flatten[:, i:i + 1] = kernel[i, :, :, :].reshape(-1, 1)
    #
    #     for i in range(img.shape[0]):
    #         col = np.zeros((out_h * out_w, C * K * K))
    #         for j in range(img.shape[1]):
    #             col_j = None
    #             for k in range(0, out_h, stride):
    #                 for l in range(0, out_w, stride):
    #                     x = img[i:i + 1, j:j + 1, (stride * k):(K + stride * k), (stride * l): (K + stride * l)]
    #                     x = x.reshape(1, -1)
    #                     if col_j is None:
    #                         col_j = x
    #                     else:
    #                         col_j = np.vstack((col_j, x))
    #             col[:, j * K * K:(j + 1) * K * K] = col_j
    #         mat = np.dot(col, kernel_flatten) + bias
    #         for j in range(result.shape[1]):
    #             result[i:i + 1, j:j + 1, :, :] = mat[:, j:j + 1].reshape((out_h, out_w))
    #
    #     return result

    def img2col(self, img, kernel, stride):
        """
        :param img: image input had been padding. dim:[N, C, H, W]
        :param kernel: convolution kernel. dim:[O, C, K, K]
        :param stride: stride
        :return: reshape of img. dim:[N * (H-K+1)/stride * (W-K+1)/stride, C * K * K]
        """
        N, C, H, W = img.shape
        O, _, K, _ = kernel.shape
        out_h = int((H - K) / stride + 1)
        out_w = int((W - K) / stride + 1)
        out_size = out_h * out_w
        x_cols = np.zeros((N * out_size, C * K * K))
        for i in range(0, H - K + 1, stride):
            i_start = i * out_w
            for j in range(0, W - K + 1, stride):
                temp = img[:, :, i:i + K, j:j + K].reshape((N, -1))
                x_cols[i_start + j::out_size, :] = temp
        return x_cols

    def conv(self, input, kernel, stride, bias):
        """
        :param input: feature map. dim:[N, C, H, W]
        :param kernel: convolution kernel. dim:[O, C, K, K]
        :param stride: stride
        :param bias: bias
        :return: convolution result
        """
        N, C, H, W = input.shape
        O, _, K, _ = kernel.shape
        out_h = int((H - K) / stride + 1)
        out_w = int((W - K) / stride + 1)
        x_cols = self.img2col(input, kernel, stride)
        weight_cols = kernel.reshape(O, -1).T
        output = np.dot(x_cols, weight_cols) + bias
        output = output.reshape((N, output.shape[0] // N, -1)).reshape((N, out_h, out_w, O))
        output = np.transpose(output, (0, 3, 1, 2))
        return output


# ---------------------------------------------------------------------------------------------------------Pooling layer
class MaxPooling:
    def __init__(self, kernel_size=2, stride=2):
        """
        :param kernel_size: the size of pooling kernel
        :param stride: stride
        """
        if kernel_size != stride:
            print('Warning: The kernel size of MaxPooling should be equal to stride. Please examine parameters.')
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        :param input: feature map. dim:[N, C, H, W]
        :return: pooling result
        """
        N, C, H, W = input.shape
        output = input.reshape(N, C, H // self.kernel_size, self.kernel_size, W // self.kernel_size, self.kernel_size)
        output = np.max(output, axis=(3, 5))
        self.mask = output.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3) != input

        # # For loop version
        # self.input = input
        # output = np.zeros((N, C, H // self.kernel_size, W // self.kernel_size))
        # for i in range(output.shape[0]):
        #     for j in range(output.shape[1]):
        #         for k in range(output.shape[2]):
        #             for l in range(output.shape[3]):
        #                 receptive_field = self.input[i:i + 1, j:j + 1,
        #                                   (self.stride * k):(self.kernel_size + self.stride * k),
        #                                   (self.stride * l): (self.kernel_size + self.stride * l)]
        #                 max_value = np.max(receptive_field)
        #                 output[i][j][k][l] = max_value

        return output

    def backward(self, eta):
        """
        :param eta: gradient that the next layer return
        :return: gradient of this layer
        """
        output = eta.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
        output[self.mask] = 0

        # # For loop version.
        # output = np.zeros_like(self.input)
        # for i in range(eta.shape[0]):
        #     for j in range(eta.shape[1]):
        #         for k in range(eta.shape[2]):
        #             for l in range(eta.shape[3]):
        #                 max = np.argmax(self.input[i:i + 1, j:j + 1, (self.stride * k):
        #                 (self.kernel_size + self.stride * k),
        #                 (self.stride * l): (self.kernel_size + self.stride * l)])
        #                 index = np.unravel_index(max, (1, 1, self.kernel_size, self.kernel_size))
        #                 output[i:i + 1, j:j + 1, (self.stride * k):(self.kernel_size + self.stride * k),
        #                 (self.stride * l): (self.kernel_size + self.stride * l)][index] = eta[i][j][k][l]

        return output


class AveragePooling:
    def __init__(self, kernel_size=2, stride=2):
        """
        :param kernel_size: the size of pooling kernel
        :param stride: stride
        """
        if kernel_size != stride:
            print('Warning: The kernel size of AveragePooling should be equal to stride. Please examine parameters.')
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        :param input: feature map. dim:[N, C, H, W]
        :return: pooling result
        """
        N, C, H, W = input.shape
        output = input.reshape(N, C, H // self.kernel_size, self.kernel_size, W // self.kernel_size, self.kernel_size)
        output = np.sum(output, axis=(3, 5))
        output = output / self.kernel_size ** 2

        # # For loop version.
        # self.input = input
        # N, C, H, W = input.shape
        # output = np.zeros((N, C, H // self.kernel_size, W // self.kernel_size))
        #
        # for i in range(output.shape[0]):
        #     for j in range(output.shape[1]):
        #         for k in range(output.shape[2]):
        #             for l in range(output.shape[3]):
        #                 receptive_field = self.input[i:i + 1, j:j + 1,
        #                                   (self.stride * k):(self.kernel_size + self.stride * k),
        #                                   (self.stride * l): (self.kernel_size + self.stride * l)]
        #                 mean_value = np.mean(receptive_field)
        #                 output[i][j][k][l] = mean_value

        return output

    def backward(self, eta):
        """
        :param eta: gradient that the next layer return
        :return: gradient of this layer
        """
        output = eta.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)

        # # For loop version.
        # output = np.zeros_like(self.input)
        # for i in range(eta.shape[0]):
        #     for j in range(eta.shape[1]):
        #         for k in range(eta.shape[2]):
        #             for l in range(eta.shape[3]):
        #                 mean_value = eta[i][j][k][l]/(self.kernel_size * self.kernel_size)
        #                 output[i:i + 1, j:j + 1, (self.stride * k):(self.kernel_size + self.stride * k),
        #                                   (self.stride * l): (self.kernel_size + self.stride * l)] += mean_value

        return output


# ---------------------------------------------------------------------------------------------------Fully Connect layer
class Fc:
    def __init__(self, in_features, out_features, bias=True):
        """
        :param in_features: input features
        :param out_features: output features
        :param bias: whether using bias
        """
        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = bias
        self.weight = np.random.randn(self.in_features, self.out_features) * (2 / self.in_features ** 0.5)
        if self.need_bias:
            self.bias = np.random.randn(out_features)
        else:
            self.bias = None

    def forward(self, input):
        """
        :param input: feature map. dim:[N, C, H, W] or [N, C*H*W]
        :return: fc result
        """
        self.input_shape = input.shape
        if input.ndim == 4:
            N = input.shape[0]
            self.input = input.reshape((N, -1))
        elif input.ndim == 2:
            self.input = input
        output = np.dot(self.input, self.weight)
        if self.need_bias:
            output += self.bias
        return output

    def backward(self, eta, learning_rate):
        """
        :param eta: gradient that the next layer return
        :param learning_rate: learning rate
        :return: gradient of this layer
        """
        delta = np.dot(eta, self.weight.T)
        delta = np.reshape(delta, self.input_shape)
        self.weight -= learning_rate * np.dot(self.input.T, eta) / self.input.shape[0]
        if self.need_bias:
            self.bias -= learning_rate * np.sum(eta, axis=0) / self.input.shape[0]

        return delta


# ---------------------------------------------------------------------------------------------Activation Function layer
class Relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, eta):
        eta[self.input <= 0] = 0
        return eta


class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-1 * input))
        return self.output

    def backward(self, eta):
        eta = eta * self.output * (1 - self.output)
        return eta


class Tanh:
    def forward(self, input):
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return self.output

    def backward(self, eta):
        eta = eta * (1 - self.output ** 2)
        return eta


# ---------------------------------------------------------------------------------------------------Loss Function layer
class Softmax_CrossEntropy:
    def forward(self, input):
        a = np.exp(input)
        b = np.repeat(np.sum(np.exp(input), axis=1, keepdims=True), input.shape[1], axis=1)
        output = a / b
        return output

    def calculate_loss(self, predicted, true):
        self.predicted = predicted
        self.true = true
        n, m = predicted.shape
        loss = np.zeros((n, 1))
        for i in range(n):
            loss[i] = -np.sum(np.multiply(true[i, :], np.log(predicted[i, :] + 1e-6)))
        loss = np.sum(loss) / n
        return loss

    def gradient(self):
        self.eta = self.predicted - self.true
        return self.eta


class MSE:
    def calculate_loss(self, predicted, true):
        self.predicted = predicted
        self.true = true
        return 0.5 * np.sum((true - predicted) ** 2)

    def gradient(self):
        self.eta = self.predicted - self.true
        return self.eta


# ---------------------------------------------------------------------------------------------Batch Normalization layer
class BN:
    def __init__(self, channel, momentum=0.9, is_train=True):
        """
        :param channel: channel number
        :param momentum: momentum coefficient
        :param is_train: whether is in training stage
        """
        self.channel = channel
        self.momentum = momentum
        self.is_train = is_train
        self.gamma = np.ones((self.channel, 1, 1))
        self.beta = np.zeros((self.channel, 1, 1))
        self.eps = 1e-5

        self.momentum_mean = np.zeros((self.channel, 1, 1))
        self.momentum_var = np.zeros((self.channel, 1, 1))

    def forward(self, input, is_train=True):
        """
         :param input: feature map. dim：[N,C,H,W]
         :return: batch normalized feature map. dim:[N,C,H,W]
         """
        self.input = input
        self.is_train = is_train
        if self.is_train:
            self.mean = np.mean(input, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
            self.var = np.var(input, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]

            if (np.sum(self.momentum_mean) == 0) and (np.sum(self.momentum_var) == 0):
                self.momentum_mean = self.mean
                self.momentum_var = self.var
            else:
                self.momentum_mean = self.momentum * self.momentum_mean + (1 - self.momentum) * self.mean
                self.momentum_var = self.momentum * self.momentum_var + (1 - self.momentum) * self.var

            self.x = (input - self.mean) / np.sqrt(self.var + self.eps)
            self.y = self.gamma * self.x + self.beta
        else:
            self.x = (input - self.momentum_mean) / np.sqrt(self.momentum_var + self.eps)
            self.y = self.gamma * self.x + self.beta
        return self.y

    def backward(self, eta, learning_rate):
        """
        :param eta: gradient that the next layer return
        :param learning_rate: learning rate
        :return: gradient of this layer
        """
        N, _, H, W = eta.shape
        gamma_grad = np.sum(eta * self.x, axis=(0, 2, 3))
        beta_grad = np.sum(eta, axis=(0, 2, 3))

        yx_grad = (eta * self.gamma)
        ymean_grad = (-1.0 / np.sqrt(self.var + self.eps)) * yx_grad
        ymean_grad = np.sum(ymean_grad, axis=(2, 3))[:, :, np.newaxis, np.newaxis] / (H * W)
        yvar_grad = -0.5 * yx_grad * (self.input - self.mean) / (self.var + self.eps) ** (3.0 / 2)
        yvar_grad = 2 * (self.input - self.mean) * np.sum(yvar_grad, axis=(2, 3))[:, :, np.newaxis, np.newaxis] / (
                    H * W)
        result = yx_grad * (1 / np.sqrt(self.var + self.eps)) + ymean_grad + yvar_grad

        self.gamma -= learning_rate * gamma_grad[:, np.newaxis, np.newaxis] / N
        self.beta -= learning_rate * beta_grad[:, np.newaxis, np.newaxis] / N

        return result
