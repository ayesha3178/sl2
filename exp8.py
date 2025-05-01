import numpy as np

class ART1:
    def __init__(self, input_size, vigilance=0.8):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []
        self.clusters = []

    def match(self, input_vector, weight):
        min_vector = np.minimum(input_vector, weight)
        return np.sum(min_vector) / np.sum(input_vector)

    def train(self, data):
        for vector in data:
            found_cluster = False
            for i, weight in enumerate(self.weights):
                if self.match(vector, weight) >= self.vigilance:
                    # Update rule (intersection)
                    self.weights[i] = np.minimum(weight, vector)
                    self.clusters[i].append(vector)
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                self.weights.append(np.copy(vector))
                self.clusters.append([vector])

    def print_clusters(self):
        for i, cluster in enumerate(self.clusters):
            print(f"\nCluster {i+1}:")
            for vec in cluster:
                print(vec)


# Example binary input vectors
input_data = np.array([
    [1, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0]
])

# Create ART1 network
art = ART1(input_size=6, vigilance=0.8)

# Train on data
art.train(input_data)

# Show resulting clusters
art.print_clusters()
