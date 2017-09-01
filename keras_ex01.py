

from keras.models import Sequential
from keras.layers import Dense
import numpy, logging

# Initiate logger
mylog = logging.getLogger()
# set logger level
mylog.setLevel('DEBUG')

mylog.debug('Generate a random seed for reproducibility')
try:
    numpy.random.seed(7)
except StandardError:
    mylog.critical('Couldn\'t generate a random seed!')

mylog.debug('Import pima indians csv datafile')
try:
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
except IOError:
    mylog.critical('Couldn\'t import pima indians data! does the file exist in the python directory?')

# Dataset is made of 9 columns. Last column is the dependent variable
#
# The below is performing a slice.
# The comma splits the slice between two dimensions - everything before the comma is operating
# on each row. Everything after the comma is operating on columns within the row. In this way,
# slide can cater to limitless data dimensions (but for practical examples we only need 2)
#
# Slices have a base of 0.
# The X variable ([:, :8]) selects every row (colon before the comma), then grabs every column until the 9th column.
# The Y variable ([:, 8] selects every row (colon before the comma), and of those rows,
# grabs all the data for the 9th column.
mylog.debug('Splitting up the input file into independent and dependent variables')
try:
    X = dataset[:, :8]
    Y = dataset[:, 8]
except StandardError:
    mylog.critical('Couldn\'t slice the data up into 9 columns!')

# This example works with the Keras sequential model. In Keras, the easiest way to add layers (according to the
# tutorial), is to define an empty Sequential() constructor, like below
mylog.debug('Define a new model from the Keras Sequential constructor')
try:
    mymodel = Sequential()
except StandardError:
    mylog.critical("Couldn't define a new model!")
# and then start adding layers to it with the .add() method. Each .add() is a new layer in the neural network.


# Here's where the magic happens - we decide the hyperparameters here, and it's hard to evaluate them.
mylog.debug("Add 3 layers to the newly defined model")
try:
    # First layer has 24 neurons and, importantly, expects 8 input variables
    mymodel.add(Dense(24, input_dim=8, activation='relu'))
    # I added a second layer, and it increased accuracy over 500 iterations by 2.5%
    mymodel.add(Dense(12, activation='relu'))
    # Third layer has only 8 neurons
    mymodel.add(Dense(8, activation='relu'))
    # Last layer has only 1 neuron - for 1 dependent variable. In this example - does the subject have diabetes or not?
    mymodel.add(Dense(1, activation='sigmoid'))
except StandardError:
    mylog.critical("Couldn't add layers to my model!")



# Compile the model, so that we can run it on our current hardware. This needs an explicit step in Keras.
mylog.debug("Compiling the model")
try:
    mymodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
except StandardError:
    mylog.critical("Couldn't compile the model!")

# Fit the model to the data, with lots of iterations
mylog.debug("Fitting the model to the data")
try:
    # Epochs is the number of iterations that we run the dataset through.
    # Batch_size is the number of "instances that are evaluated before a weight update is performed", whatever
    # that means.
    # Verbosity is the level of detail about the model's evaluation that's printed to the console. 
    # A higher number is less detail.
    mymodel.fit(X, Y, epochs=500, batch_size=10,  verbose=1)
    # mymodel.fit(X, Y, epochs=500, batch_size=10,  verbose=2)
except StandardError:
    mylog.critical("Couldn't fit the model to the data!")

# Score the model's output against the data
mylog.debug("Evaluating the model output against the data")
try:
    # Evaluate command is where the work happens
    scores = mymodel.evaluate(X, Y)
    # Print the results of the evaluation.
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
except StandardError:
    mylog.critical("Couldn't evaluate the model!")


# calculate predictions
mylog.debug("Evaluating the model output against the data")
try:
    # Predict outcomes based on the predict function? Not sure what this does - look it up later!
    predictions = mymodel.predict(X)
except StandardError:
    mylog.critical("Couldn't create predictions against the model!")

