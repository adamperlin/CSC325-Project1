from locale import normalize
import tensorflow as tf
import tensorflow_datasets as tfds

def load():
    (ds_train, ds_test), ds_info = tfds.load('mnist', 
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True)
    
    return (ds_train, ds_test, ds_info)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def preprocess(ds_train, ds_test, ds_info):
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)

    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_test)

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(0.02), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

model = build_model()
print(model.summary())

train, test, info = load()
(ds_train, ds_test) = preprocess(train, test, info)


model.fit(ds_train, epochs=20, validation_data=ds_test)


