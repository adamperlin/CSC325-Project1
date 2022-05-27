"""
CSC 325-03 Project 1

"""

import collections
import random

import nest_asyncio
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import argparse
import tensorflow_datasets as tfds

# =========== Configuration ==========
NUM_CLIENTS = 10
NUM_EPOCHS = 20
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_TRAINING_ROUNDS = 50


# ================ Helper Functions =============================
def _batch_format(elm):
    return collections.OrderedDict(
        x=tf.reshape(elm['pixels'], [-1, 784]),
        y=tf.reshape(elm['label'], [-1, 1])
    )


def preprocess(dataset):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1) \
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
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        print(f"ypred: {len(y_pred)}")
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)


def get_iterative_process(model_fn):
    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )


def train_model(iterative_process, ds_train, num_rounds, num_clients):
    logdir = "/tmp/logs/scalars/training/"
    summary_writer = tf.summary.create_file_writer(logdir)
    state = iterative_process.initialize()

    # run one round of training
    # state, metrics = iterative_process.next(state, federated_training_data)
    # print(f"round 1, metrics={metrics}")

    # run {11} rounds of training, logging output
    with summary_writer.as_default():
        for round_num in range(num_rounds):
            train_data = get_data_for_clients(ds_train, num_clients)
            # NOTE: the key observation here is that the loss parameter in the model
            #   decreases with each iteration, which indicates convergence => i.e. the goal
            state, metrics = iterative_process.next(state, train_data)
            print(metrics['train'])

            # print metric data to summary
            for name, value in metrics['train'].items():
                tf.summary.scalar(name, value, step=round_num)

    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', help="number of training rounds to perform", default=NUM_TRAINING_ROUNDS,
                        type=int)
    parser.add_argument('--num_users', help="number of users to distribute training across", default=NUM_CLIENTS,
                        type=int)

    return parser.parse_args()


def get_data_for_clients(ds_train, num_clients):
    sample_clients = random.sample(ds_train.client_ids, k=num_clients)
    return make_federated_data(ds_train, sample_clients)


def evaluate(test_data, evaluation, state):
    trials = 20
    avg_acc = 0.0
    for i in range(trials):
        test_metrics = evaluation(state.model, get_data_for_clients(test_data, 10))
        avg_acc += test_metrics['eval']['sparse_categorical_accuracy']

    avg_acc /= trials

    print(f"Average categorical accuracy ({trials} user sets): {avg_acc}")


def load_centralized():
    (ds_train, ds_test), ds_info = tfds.load('mnist',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)

    return ds_train, ds_test, ds_info


def preprocess_centralized(ds_train, ds_test, ds_info):
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.batch(128)
    ds_test = ds_test.map(lambda img, label: (tf.reshape(img, [-1, 784]), tf.reshape(label, [-1, 1])))
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    print(ds_test)

    return ds_train, ds_test


def main():
    # run necessary configuration tasks
    setup()
    args = parse_args()

    # Load the emnist data set
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    cmnist_train, cmnist_test, cmnist_info = load_centralized()
    _, cmnist_processed_test = preprocess_centralized(cmnist_train, cmnist_test, cmnist_info)

    # create a dataset
    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    preprocessed_dataset = preprocess(example_dataset)

    def _get_model():
        model = create_keras_model()
        return tff.learning.from_keras_model(
            model,
            input_spec=preprocessed_dataset.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    iterative_process = get_iterative_process(_get_model)
    # print(iterative_process.initialize.type_signature.formatted_representation())

    state = train_model(iterative_process, emnist_train, args.num_rounds, args.num_users)
    # Model is trained

    # Start evaluation of the trained model
    evaluation = tff.learning.build_federated_evaluation(_get_model)
    evaluate(emnist_test, evaluation, state)


if __name__ == '__main__':
    main()
