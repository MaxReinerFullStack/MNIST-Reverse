import neural_net
import mnist_loader
import time
import os.path

if os.path.isfile("../data/weights_file1"):
	network = neural_net.Net([784, 20, 30, 10])
	network.load()
	print("Network loaded.")
else:

	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	#network = neural_net.Net([784, 30, 10])
	network = neural_net.Net([784, 20, 30, 10])
	network.load()
	print("Training network...")
	# Training takes O(n^2) time now (a really long time for the whole data set) because I wanted to be very explicit about how errors and weights were being calculated.
	start = time.time()
	network.train(training_data, 3.0)
	end = time.time()
	print("Training time:")
	print(end - start)
	print("Test network...")
	start = time.time()
	network.test(test_data)
	end = time.time()
	print("Test time:")
	print(end - start)

	network.save()
	print("Network successfully saved.")

# Imagine the input of an output using imagine()

# This is like asking network the question
# "Hey, I have trained you to be able to classify a 3. Now, I'm curious, what do you think a 3 looks like?".

network.imagine(3)

