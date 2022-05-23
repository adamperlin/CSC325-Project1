"""
CSC 325-03 Project 1

"""

import collections
import nest_asyncio
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np


def main():
    np.random.seed(0)
    nest_asyncio.apply()

    federated_result = tff.federated_computation(lambda: 'Hello, World!')()
    print(federated_result)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
