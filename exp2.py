# McCulloch-Pitts Neural Net for AND-NOT Function

def mcp_neuron(inputs, weights, threshold):
    # Calculate weighted sum
    total = sum(i * w for i, w in zip(inputs, weights))
    # Apply step function
    return 1 if total >= threshold else 0

# Define all combinations of inputs A and B
input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Define weights for A and B: A has positive weight, B has negative weight (NOT B)
weights = [1, -1]
threshold = 1

# Print header
print("A B | A AND (NOT B)")
print("---------------------")

# Compute output for each input combination
for inputs in input_combinations:
    output = mcp_neuron(inputs, weights, threshold)
    print(f"{inputs[0]} {inputs[1]} | {output}")
