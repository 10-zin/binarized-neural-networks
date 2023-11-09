import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist


np.random.seed(0)
NUM_CLASSES = 10
BATCH_SIZE = 512
EPOCHS = 10


def init_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Normalizing the data makes it easier for the model to arrive at
    # the optimal value. Since our matrix values can be between 0 - 255,
    # we divide by 255.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # The data that we have is of the shape (60000 * 28 * 28), but we don't want to pass a 28 X 28 image to the neural network.
    # We want to pass this image flattened out as a single vector of size 784 X 1. We can use numpy reshape operation to do this.
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Defining the architecture of the model.
    model = Sequential()
    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model Summary")
    print(model.summary())

    # Train the model
    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    return model


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print()
    print()
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")


def main():
    X_train, X_test, y_train, y_test = init_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
