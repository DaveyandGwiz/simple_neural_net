import numpy as np

# Introduction: In the brain, neurons are the fundamental units that transmit signals.
# Each neuron receives inputs from other neurons, processes those signals, and transmits an output to other neurons.
# In an artificial neural network, the same process is simulated using mathematical functions.

# A simple dataset of 3x3 "images" and their labels (0: no line, 1: horizontal line)
training_inputs = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # No line
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # Horizontal line
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Horizontal line
    [0, 0, 0, 0, 0, 0, 1, 1, 1],  # Horizontal line
    [1, 0, 0, 1, 0, 0, 1, 0, 0],  # No line
    [1, 0, 0, 0, 0, 0, 0, 0, 1]   # No line
])

# Labels for the training set (0: no line, 1: horizontal line)
training_labels = np.array([[0], [1], [1], [1], [0], [0]])

# Explanation: Just like neurons in the brain receive signals from multiple neurons,
# our artificial neurons will receive multiple inputs (3x3 pixel grids).
# These inputs will be processed to "decide" if the image contains a horizontal line or not.

# Seed the random number generator for reproducibility
np.random.seed(1)

# In neural networks, the initial weights are set randomly, similar to how genetics give organisms a predisposition.
# Just like how some individuals may have a natural talent for certain activities,
# some neural networks may be initially "better" at certain tasks depending on their starting weights.
# These initial random weights give each neural network a unique "starting point" for learning.
weights = 2 * np.random.random((9, 1)) - 1

# Sigmoid activation function: This function models the "activation" of a neuron in the brain.
# In the brain, a neuron activates and fires an output when the combined input signals reach a certain threshold.
# The sigmoid function simulates this by producing a value between 0 and 1 based on the input.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function: Used in the learning process to adjust weights.
# In the brain, neurons adjust their connections based on the strength of the signal received.
# The derivative helps us calculate how much to adjust these connections (weights) in our neural network.
def sigmoid_derivative(x):
    return x * (1 - x)

# Train the network for 10 iterations
# In the brain, learning happens through repeated exposure to stimuli.
# In our artificial neural network, the weights (connections) are adjusted repeatedly (10 times here) as it learns from the data.
for iteration in range(10):
    # Forward pass: The input signals are passed through the network, just like in the brain,
    # where signals pass through neurons and synapses.
    input_layer = training_inputs
    # Here, we calculate the neuron's output by multiplying inputs with weights and applying the sigmoid function (activation).
    output = sigmoid(np.dot(input_layer, weights))

    # Calculate the error (difference between the predicted output and the actual output)
    # In the brain, if the predicted response (output) is incorrect, learning occurs by adjusting synaptic strengths.
    error = training_labels - output

    # Backpropagation: Adjust the weights
    # In the brain, this would be analogous to neurons strengthening or weakening their connections based on the error.
    # The sigmoid_derivative helps us calculate the sensitivity of the neuron to changes in input, similar to how neurons learn from experience.

    # We calculate adjustments this way because we want to change the weights in proportion to:
    # 1. How large the error is (big errors require bigger corrections).
    # 2. How much influence the input had on the output (inputs that contributed more to the error get larger adjustments).
    # 3. How sensitive the output is to changes in the input (using the sigmoid derivative to ensure the adjustments are scaled properly).

    adjustments = np.dot(input_layer.T, error * sigmoid_derivative(output))

    # does backprogation occur in the brain?
    """
    Timothy P. Lillicrap et al. Nature, (Backpropagation and the brain)
    During learning, the brain modifies synapses to improve behaviour. 
    In the cortex, synapses are embedded within multilayered networks, making it difficult to determine the effect of an individual synaptic modification on the behaviour of the system. 
    The backpropagation algorithm solves this problem in deep artificial neural networks, 
    but historically it has been viewed as biologically problematic. 
    Nonetheless, recent developments in neuroscience and the successes of artificial neural networks have reinvigorated interest in whether backpropagation offers insights for understanding learning in the cortex. 
    """

    # Update the weights: This step mimics how the brain learns by adjusting synaptic connections.
    # In our artificial neural network, weights are updated so that the output moves closer to the actual label.
    weights += adjustments

    # Printing the error at each iteration to show the learning process in action.
    print(f"Iteration {iteration + 1}, Error: {error}")

# Test the neural network with a new "image" (a horizontal line)
# In the brain, after learning, neurons respond better to familiar stimuli. Here, we present a new image (stimulus) to see how well our neural network has learned.
new_image = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

# The network uses the learned weights to predict whether this new image contains a horizontal line.
# In the brain, this would be similar to a neuron recognizing a familiar pattern.
result = sigmoid(np.dot(new_image, weights))

#


# Output the network's prediction: This is like the brain's decision-making process after interpreting signals.
print(f"Prediction for the new image: {result}")
print("Class: Horizontal line" if result > 0.5 else "Class: No line")

# Explanation: In this last step, the neural network compares the result to the threshold of 0.5,
# just like a neuron in the brain firing when the signal strength exceeds a certain threshold.
# If the result is above 0.5, the network predicts that the image contains a horizontal line.
