
from tensorflow import keras, optimizers
import ipdb


class KerasTrainingModule:
    def __init__(self, classifier:keras.Model, lr: float = 1e-3, epochs:int=10, loss="binary_crossentropy", metrics="accuracy"):
        self.classifier = classifier
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        

    def compile(self):
        self.classifier.compile(optimizer=keras.optimizers.Adam(1e-3), loss=self.loss)
        print("Compiled successfully!")


    def fit(self, train_ds, val_ds):
        callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        ]
        self.classifier.fit(
            train_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=val_ds,
        )
