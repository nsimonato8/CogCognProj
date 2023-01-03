import tensorflow as tf


# https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/

class CNN:
    def __init__(self, model, optimizer='', loss_fn=None, device=None):
        super().__init__()
        self.optimizer = optimizer
        self.model = model

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
