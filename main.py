"""
CSC 325-03 Project 1

"""

import collections
import nest_asyncio
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np


# =========== Configuration ==========
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_TRAINING_ROUNDS = 11


# ================ Helper Functions =============================
def _batch_format(elm):
    return collections.OrderedDict(
        x=tf.reshape(elm['pixels'], [-1, 784]),
        y=tf.reshape(elm['label'], [-1, 1])
    )


def preprocess(dataset):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1)\
        .batch(BATCH_SIZE).map(_batch_format).prefetch(PREFETCH_BUFFER)


def setup():
    np.random.seed(0)
    nest_asyncio.apply()


def showPlot():
    """
    Show a pyplot chart in a standard manner. Add all configurations here
    :return: Nothing
    """

    plt.grid(False)
    plt.show()
# ==================================================================


# ======== These functions are for tutorial purposes only ===========
def sanity_check():
    # Sanity check, not project crucial
    federated_result = tff.federated_computation(lambda: 'Hello, World!')()
    print(federated_result)


def showClientIds(test_data):
    client_ids = test_data.client_ids
    print(f"There are {len(client_ids)} client ids.")


def showPreprocessedData(dataset):
    preprocessed = preprocess(dataset)
    sample = tf.nest.map_structure(lambda x: x.numpy(),
                                   next(iter(preprocessed)))
    print(sample)
# ====================================================================


# ===================== Plot demos ===================================
def demo1(dataset):
    # Show one element
    example_element = next(iter(dataset))
    print(example_element['label'].numpy())  # 1

    # plot training data - 1 item from 1 clients sample
    plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
    showPlot()


def demo2(dataset):
    # plot training data - 50 items
    figure = plt.figure(figsize=(20, 4))
    j = 0

    for example in dataset.take(50):
        plt.subplot(5, 10, j + 1)
        plt.imshow(example["pixels"].numpy(), cmap='gray', aspect='equal')
        plt.axis("off")
        j += 1

    showPlot()


def demo3(training_data):
    # Number of examples per layer for a sample of clients
    f = plt.figure(figsize=(12, 7))
    f.suptitle('Label Counts for a Sample of Clients')
    for i in range(6):
        client_dataset = training_data.create_tf_dataset_for_client(
            training_data.client_ids[i])
        plot_data = collections.defaultdict(list)
        for example in client_dataset:
            # Append counts individually per label to make plots
            # more colorful instead of one color per plot.
            label = example['label'].numpy()
            plot_data[label].append(label)
        plt.subplot(2, 3, i + 1)
        plt.title('Client {}'.format(i))
        for j in range(10):
            plt.hist(
                plot_data[j],
                density=False,
                bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    showPlot()


def demo4(training_data):
    # mean image per label for 3 clients
    for i in range(3):
        client_data = training_data.create_tf_dataset_for_client(
            training_data.client_ids[i])
        plot_data = collections.defaultdict(list)
        for sample in client_data:
            plot_data[sample['label'].numpy()].append(sample['pixels'].numpy())
        f = plt.figure(i, figsize=(12, 5))
        f.suptitle(f"Client {i}'s Mean Image per Label")

        # show each digit
        for j in range(10):
            mean = np.mean(plot_data[j], 0)
            plt.subplot(2, 5, j + 1)
            plt.imshow(mean.reshape((28, 28)))
            plt.axis('off')

        showPlot()
# ============================================================


# =================== main project methods ==========================
def make_federated_data(client_data, ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in ids
    ]


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784, )),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


def get_iterative_process(dataset):
    preprocessed_dataset = preprocess(dataset)

    def _get_model():
        model = create_keras_model()
        return tff.learning.from_keras_model(
            model,
            input_spec=preprocessed_dataset.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    return tff.learning.build_federated_averaging_process(
        model_fn=_get_model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )


def main():
    # run necessary configuration tasks
    setup()

    # Load the emnist data set
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    # create a dataset
    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

    # TODO: implement random sampling for better fitting
    federated_training_data = make_federated_data(emnist_train, sample_clients)

    print(f'Number of client datasets: {len(federated_training_data)}')
    print(f'First dataset: {federated_training_data[0]}')

    iterative_process = get_iterative_process(example_dataset)
    # print(iterative_process.initialize.type_signature.formatted_representation())

    state = iterative_process.initialize()

    # run one round of training
    # state, metrics = iterative_process.next(state, federated_training_data)
    # print(f"round 1, metrics={metrics}")

    # run {11} rounds of training
    for round_num in range(NUM_TRAINING_ROUNDS):
        # NOTE: the key observation here is that the loss parameter in the model
        #   decreases with each iteration, which indicates convergence => i.e. the goal
        state, metrics = iterative_process.next(state, federated_training_data)
        print(f"Round {round_num:2d}, metrics={metrics}")

    # Model is trained


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
