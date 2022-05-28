from locale import normalize
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

def evaluation_summary(model, test, info):
    df = tfds.as_dataframe(test, info)
    y_pred = np.argmax(model.predict(test), axis=1)
    print(len(y_pred))
    y_true = np.concatenate(df['label'])
    print(y_true)
    print(y_pred)
    #print(df['label'][0])
    #y_true = np.array(df['label'])

    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return (report, conf_matrix)