import numpy as np


class BAM:

    def __init__(self, input_size, output_size):

        self.weights = np.zeros((input_size, output_size))

    def train(self, input_patterns, output_patterns):

        for x, y in zip(input_patterns, output_patterns):
            self.weights += np.outer(x, y)

        print("Updated Weight Matrix:\n", self.weights)

    def recall(self, input_pattern, direction="forward"):

        if direction == "forward":

            output = np.sign(np.dot(input_pattern, self.weights))

        else:

            output = np.sign(np.dot(input_pattern, self.weights.T))

        return output

    # Define two pairs of bipolar vectors (1 and -1 instead of 0 and 1)


input_patterns = np.array([[1, -1, 1], [-1, 1, -1]])

output_patterns = np.array([[1, -1], [-1, 1]])

# Initialize BAM

bam = BAM(input_size=3, output_size=2)

# Train BAM with the given patterns

bam.train(input_patterns, output_patterns)

# Recall from input to output

test_input = np.array([1, -1, 1])

retrieved_output = bam.recall(test_input, direction="forward")

print("\nRecalled Output for input {}: {}".format(test_input, retrieved_output))

# Recall from output to input

test_output = np.array([1, -1])

retrieved_input = bam.recall(test_output, direction="backward")

print("\nRecalled Input for output {}: {}".format(test_output, retrieved_input))
