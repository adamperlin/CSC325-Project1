"""
CSC 325-03 Project 1

"""

import nest_asyncio
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np


def main():
    np.random.seed(0)
    nest_asyncio.apply()

    tff.federated_computation(lambda: 'Hello, World!')()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
