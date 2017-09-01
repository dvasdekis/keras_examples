

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
X = dataset[:, :8]
Y = dataset[:, 8]
