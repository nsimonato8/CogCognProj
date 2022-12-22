import tensorflow as tf
from tensorflow.keras import layers, models


# https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/

class CNN:
    def __init__(self, input_shape, n_classes=26, activation='relu', optimizer='', loss_fn=None, learning_rate=0.1,
                 momentum=0.9, device=None):
        super().__init__()
        self.optimizer = optimizer
        self.model = models.Sequential(layers=[
            layers.Conv2d(filter=2, input_shape=input_shape, activation=activation, kernel_size=(3, 3),
                          padding='valid'),
            layers.MaxPooling2d(kernel_size=(2, 2)),

            layers.Conv2d(filter=64, activation=activation, kernel_size=(3, 3), padding='valid'),
            layers.MaxPooling2d(kernel_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(64),
            layers.Dense(n_classes)])

        if loss_fn is None:
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        if device is not None:
            self.model.to(device)
        pass

    def train_CNN(self, num_epoch, train_ds, test_ds, batch_size=32):
        self.model.fit(x=train_ds.data,
                       y=train_ds.targets,
                       epochs=num_epoch,
                       batch_size=batch_size,
                       validation_data=(test_ds.data, test_ds.targets),
                       verbose=2)
