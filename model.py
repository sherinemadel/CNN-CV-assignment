import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pickle

train_file_path = 'cifar-100-python/train'
test_file_path = 'cifar-100-python/test'
meta_file_path = 'cifar-100-python/meta'

train_size = 50000
test_size = 10000
batch = 100
width = 32
height = 32
depth = 3

tensor_images = tf.placeholder(tf.float32, [None, width, height, depth])
tensor_labels = tf.placeholder(tf.int32, [None])
one_hot = tf.one_hot(tensor_labels, depth=10)


def unpickle(file, size):
    #loding the dataset
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
        images = dictionary[b'data']
        images = images.reshape(size, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        labels = dictionary[b'fine_labels']
        labels = np.array(labels)
    return images, labels


def visualize(images):
    # Visualizing 5x5 random CIFAR 100
    fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(images)))
            print(len(images))
            print(i)
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(images[i:i + 1][0])
    plt.show()
    # Visualizing 1 CIFAR 100
    # plt.imshow(images[0], interpolation='nearest')
    # plt.show()


def conv2d(images, kernel):
    return tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(norm):
    return tf.nn.max_pool(norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_kernel(specs):
    kernel = tf.variable(tf.truncated_normal(specs, stddev=0.1))
    return tf.Variable(kernel)


def create_bias(specs):
    bias = tf.constant(0.1, shape=specs)
    return tf.Variable(bias)


def build_model(images):
    # 1st layer
    kernel1 = create_kernel([7, 7, 1, 32])
    bias1 = create_bias([32])
    conv1 = conv2d(images, kernel1)
    pre_act1 = tf.nn.bias_add(conv1, bias1)
    relu1 = tf.nn.relu(pre_act1)
    pool1 = max_pool(relu1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    # 2nd layer
    kernel2 = create_kernel([5, 5, 2, 64])
    bias2 = create_bias([64])
    conv2 = conv2d(norm1, kernel2)
    pre_act2 = tf.nn.bias_add(conv2, bias2)
    relu2 = tf.nn.relu(pre_act2)
    pool2 = max_pool(relu2)
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 3nd layer
    kernel3 = create_kernel([3, 3, 3, 128])
    bias3 = create_bias([128])
    conv3 = conv2d(norm2, kernel3)
    pre_act3 = tf.nn.bias_add(conv3, bias3)
    relu3 = tf.nn.relu(pre_act3)
    pool3 = max_pool(relu3)
    norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 4th layer
    kernel4 = create_kernel([1, 1, 4, 256])
    bias4 = create_bias([256])
    conv4 = conv2d(norm3, kernel4)
    pre_act4 = tf.nn.bias_add(conv4, bias4)
    relu4 = tf.nn.relu(pre_act4)
    pool4 = max_pool(relu4)
    norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # MLP layers
    kernel5 = create_kernel([8 * 8 * 64, 64])
    bias5 = create_bias([64])
    pool5 = tf.reshape(norm4, [-1, 8 * 8 * 64])
    relu5 = tf.nn.relu(tf.matmul(pool5, kernel5) + bias5)

    kernel6 = create_kernel([64, 10])
    bias6 = create_bias([10])

    model = tf.nn.softmax(tf.matmul(relu5, kernel6) + bias6)
    return model


def cost_function(model, labels):
    cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(model, 1e-10, 1e20)))
    train_step = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, cross_entropy, accuracy


def main():
    train_images, train_labels = unpickle(train_file_path, train_size)
    test_images, test_labels = unpickle(test_file_path, test_size)
    test_labels = tf.one_hot(test_labels, depth=100)

    visualize(train_images)

    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16))
    sess.run(init)

    #looping on each batch of 100 train images
    index = 0
    row = []
    for b in range(batch):

        print("batch", b)
        avg_cost = 0
        for j in range(int(train_images.shape[0] / batch)):

            subset = range((j * batch), ((j + 1) * batch))
            data = train_images[subset, :, :, :]
            labels = tf.one_hot(train_labels[subset], depth=100)

            model = build_model(data)

            train_step, cross_entropy, accuracy = cost_function(model, labels)

            avg_cost += [train_step, cross_entropy] / data.shape[0]

            index = index + 1

            if index % 10 == 0:
                model_test = build_model(test_images)
                train_step1, cross_entropy1, accuracy1 = cost_function(model_test, test_labels)
                row.append(accuracy1)

        print(row[-1])


#run the code
main()
