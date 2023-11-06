import tensorflow as tf
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np

import utilities


def show_9_images(data):
	for i in range(9):
		# define subplot
		# arg1 = rows, arg2 = cols, arg3 = position (1-n where n = row*col)
		# nrows * ncols defines how many items we can fit
		plt.subplot(3, 3, i + 1)
		# plot raw pixel data
		# reshape data here
		reshaped_data = tf.reshape(data[i], shape=[28, 28])
		plt.imshow(reshaped_data, cmap=plt.get_cmap('gray'))
	# show the figure
	plt.show()


def show_custom_test_images(data):
	for i in range(30):
		# define subplot
		# arg1 = rows, arg2 = cols, arg3 = position (1-n where n = row*col)
		# nrows * ncols defines how many items we can fit
		plt.subplot(3, 10, i + 1)
		# plot raw pixel data
		# reshape data here
		reshaped_data = tf.reshape(data[i], shape=[28, 28])
		plt.imshow(reshaped_data, cmap=plt.get_cmap('gray'))
	# show the figure
	plt.show()


def load_data():
	mndata = MNIST('./images')

	(training_images, training_labels) = mndata.load_training()
	(test_images, test_labels) = mndata.load_testing()
	training_images = np.asarray(training_images)
	training_labels = np.asarray(training_labels)
	test_images = np.asarray(test_images)
	test_labels = np.asarray(test_labels)
	print(
		'Training data shape: data=%s, labels=%s' % (training_images.shape, training_labels.shape))
	print('Training data shape: data=%s, labels=%s' % (test_images.shape, test_labels.shape))
	return training_images, training_labels, test_images, test_labels


def normalize_data(data):
	return (data / 255) - 0.5


@tf.autograph.experimental.do_not_convert
def define_model():
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(256, (3, 3), input_shape=(28, 28, 1), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="SAME"))
	model.add(tf.keras.layers.MaxPool2D(2))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(10, activation='relu'))
	model.add(tf.keras.layers.Dense(24, activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	# compile model
	sgd_optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	model.summary()
	return model


@tf.autograph.experimental.do_not_convert
def fit(model,
        x,
        y,
        shuffle=True,
        epochs=3,
        batch_size=64,
        validation_split=0.4):
	return model.fit(x, y, shuffle=shuffle, epochs=epochs, batch_size=batch_size,
	                 validation_split=validation_split)


def plot_model_statistics(scores, histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
	# print summary
	print('Accuracy: mean=%ds.3, n=%d' % (tf.reduce_mean(scores) * 100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()


def test_custom_data_model(model):
	custom_data = utilities.load_custom_test_data()
	show_custom_test_images(custom_data)
	custom_data = normalize_data(custom_data)
	custom_data = tf.reshape(custom_data, shape=[-1, 28, 28, 1])
	predictions = model.predict(
		custom_data,
		batch_size=None,
		verbose='auto'
	)
	predictions = np.argmax(predictions, axis=1)
	for i in range(3):
		print(predictions[i*10:i*10+10])


@tf.autograph.experimental.do_not_convert
def main(test_model=False):
	if not test_model:
		training_data, training_labels, test_data, test_labels = load_data()
		show_9_images(training_data)
		training_data, test_data = normalize_data(training_data), normalize_data(test_data)
		model = define_model()
		scores, histories = list(), list()
		# reshape data
		training_data = tf.reshape(training_data, shape=[-1, 28, 28, 1])
		test_data = tf.reshape(test_data, shape=[-1, 28, 28, 1])
		# fit model
		history = fit(model, x=training_data, y=training_labels)
		# evaluate model
		_, acc = model.evaluate(
			x=test_data,
			y=test_labels)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
		# save the model
		model.save('cnn.keras')
		# test the model on custom data
		test_custom_data_model(model)
		# visualise data
		plot_model_statistics(scores, histories)
	else:
		# load the model
		model = tf.keras.models.load_model('cnn.keras')
		print("loaded model")
		test_custom_data_model(model)


main(test_model=True)
