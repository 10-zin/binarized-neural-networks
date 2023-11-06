import cv2
import os
import numpy as np


COLOR_FOLDER = "Test_data/colorful"
CURLY_FOLDER = "Test_data/curly"
SIMPLE_FOLDER = "Test_data/simple"


def rgb2gray(rgb):
	# green channel works for most cases as a grayscale proxy
	return rgb[:, :, 1]


def resize_to_mnist_image_size(folder_name, source_filename):
	img = cv2.imread(folder_name + source_filename)
	res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
	return res


def load_custom_test_data(visualise=False):
	colorlist = os.listdir(COLOR_FOLDER)

	custom_test_data = []

	for file in colorlist:
		if file.endswith(".png"):
			# color image
			resized_image = resize_to_mnist_image_size(COLOR_FOLDER + "/", file)
			grayscale_image = rgb2gray(resized_image)
			if visualise:
				cv2.imshow('color photo grayscale resized', grayscale_image)
				cv2.waitKey(100)
				cv2.destroyAllWindows()
			custom_test_data.append(grayscale_image)
			# curly image
			resized_image = resize_to_mnist_image_size(CURLY_FOLDER + "/", file)
			grayscale_image = rgb2gray(resized_image)
			if visualise:
				cv2.imshow('curly photo grayscale resized', grayscale_image)
				cv2.waitKey(100)
				cv2.destroyAllWindows()
			custom_test_data.append(grayscale_image)
			# simple image
			resized_image = resize_to_mnist_image_size(SIMPLE_FOLDER + "/", file)
			grayscale_image = rgb2gray(resized_image)
			if visualise:
				cv2.imshow('simple photo grayscale resized', grayscale_image)
				cv2.waitKey(100)
				cv2.destroyAllWindows()
			custom_test_data.append(grayscale_image)

	return np.array(custom_test_data)


load_custom_test_data()
