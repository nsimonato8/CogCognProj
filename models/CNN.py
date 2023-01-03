import tensorflow as tf


# https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/

class CNN:
    def __init__(self, model, learning_rate=0.01, optimizer=None, loss_fn=None, device=None):
        super().__init__()
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer

        self.model = model

        if loss_fn is None:
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        # if device is not None:
        #     self.model.to(device)
        pass

    def train_CNN(self, num_epoch, train_ds, test_ds, batch_size=32):
        self.model.fit(x=train_ds[0],
                       y=train_ds[1],
                       epochs=num_epoch,
                       batch_size=batch_size,
                       validation_data=(test_ds[0], test_ds[1]),
                       verbose=2)
