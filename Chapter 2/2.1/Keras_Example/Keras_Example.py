#################################################################################################
# Keras implementation of the simple NN that classifies bundle stability (cf. Section 2.1)      #
#################################################################################################
from __future__ import print_function  # for backwards compatibility: uses print() also in python2

from keras.models import Sequential  # feed foorward NN
from keras.layers import Dense  # need only fully connected (dense) layers
from keras import optimizers  # need only fully connected (dense) layers
import numpy as np

import matplotlib as mpl  # for plotting
mpl.use('TkAgg')  # needed on mac with virtual environment
import matplotlib.pyplot as plt  # for plotting
from mpl_toolkits.mplot3d import Axes3D  # for plotting

# Optional: Seed the random number generator for reproducibility
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

#################################################################################################
# Read in the full data set                                                                     #
#################################################################################################
# This step is different for every training set. In the case at hand, the training data is just a .txt file with an array
# (It is an array of arrays of the form
# [[[x_0^0,x_1^0], y^0], [[x_0^1,x_1^1], y^1], ...]
# The [x_0^i, x_1^i] are integers and the y^i are labels (0: unstable, 1: stable)
hnd = open("../stability_data.txt", "r")
all_data = eval(hnd.read())


#################################################################################################
# perform a train:test split                                                                    #
#################################################################################################
# shuffle the entire data set once to get random train and test pairs
rng.shuffle(all_data)

# perform a train:test split of 80:20
split_point = int(len(all_data)*0.8)
x_train, x_test = [x[0] for x in all_data[0:split_point]], [x[0] for x in all_data[split_point+1:]]
y_train, y_test = [[x[1]] for x in all_data[0:split_point]], [[x[1]] for x in all_data[split_point+1:]]


#################################################################################################
# Define the NN hyperparameters                                                                 #
#################################################################################################
# number of nodes in each layer
input_dim = 2
hidden1_dim = 4
hidden2_dim = 4
output_dim = 1

# set training variables
epochs = 501
batch_size = 32
learning_rate = 0.01


#################################################################################################
# Set up and initialize the NN                                                                  #
#################################################################################################
# create the NN
nn = Sequential()
nn.add(Dense(hidden1_dim, activation='sigmoid', input_dim=input_dim))
nn.add(Dense(hidden2_dim, activation='sigmoid'))  # automatically infer input dimensions
nn.add(Dense(output_dim,  activation='sigmoid'))  # automatically infer input dimensions


#################################################################################################
# Specify the optimizer                                                                         #
#################################################################################################
adam = optimizers.adam(lr=learning_rate)  # use ADAM optimizer
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # mean_squared_error works as well
nn.summary()

#################################################################################################
# Train the NN                                                                                  #
#################################################################################################
nn_output = nn.fit(np.array(x_train), np.array(y_train), epochs=epochs, batch_size=batch_size, verbose=2 , validation_data=(np.array(x_test), np.array(y_test)))
print("\nTraining complete!")


#################################################################################################
# Plot the loss during training                                                                 #
#################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(nn_output.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_loss.pdf", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


#################################################################################################
# Evaluate the NN                                                                               #
#################################################################################################

# find predictions on test set
pred = nn.predict(np.array(x_test))

acc = 0
for i in range(len(y_test)):
    if round(pred[i]) == y_test[i][0]:
        acc += 1

print("Validation Accuracy: {0:.2f}".format(float(acc)/float(len(y_test))))

for i in range(len(x_test)):
    print("{}:  \t(Actual, Prediction): ({}, {})".format(x_test[i], y_test[i][0], int(round(pred[i][0]))))

print("\nEvaluation complete!")


#################################################################################################
# Plot prediction of NN on all data                                                             #
#################################################################################################
x_all = [e[0] for e in all_data]
pred = nn.predict(np.array(x_all))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([e[0][0] for e in all_data], [e[0][1] for e in all_data], [e[0] for e in pred])
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('y')

# plt.show()
plt.savefig("./example_function.pdf", dpi=300, bbox_inches='tight')
plt.close()
