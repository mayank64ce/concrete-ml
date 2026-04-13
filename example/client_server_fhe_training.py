import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import SGDClassifier
from concrete import fhe

# Generate synthetic dataset for classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model for training with Fully Homomorphic Encryption (FHE)
model = SGDClassifier(fit_encrypted=True, parameters_range=(-1, 1))

# Train the model on the unencrypted data (FHE is disabled for this part)
# This step initializes the FHE circuit
model.fit(X_train, y_train, fhe="disable")

# Extract the FHE circuit generated during training
circuit = model.training_quantized_module.fhe_circuit

def calculate_accuracy(predictions, labels):
    """Calculate the accuracy of predictions against the true labels."""
    correct_predictions = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct_predictions / len(labels)
    return accuracy

def make_predictions(X, weights, bias):
    """Make predictions using the updated weights and bias from the model."""
    predictions = np.dot(X, weights[0]) + bias[0]
    logits = 1 / (1 + np.exp(-predictions))
    binary_predictions = (logits > 0.5).astype(int)
    return binary_predictions.flatten()

# Save the FHE circuit to a file to be used by the server for encrypted computations
circuit.server.save("server.zip")

# ----- Server-Side Operations -----
# Load the FHE circuit on the server side from the saved file
server = fhe.Server.load("server.zip")

# Serialize the client specifications to enable client-side encryption/decryption setup
serialized_client_specs = server.client_specs.serialize()

# ----- Client-Side Operations -----
# Deserialize the client specifications received from the server
client_specs = fhe.ClientSpecs.deserialize(serialized_client_specs)

# Initialize the client with the deserialized specifications
client = fhe.Client(client_specs)

# Generate the necessary encryption/decryption keys on the client side
client.keys.generate()

# Serialize the evaluation keys to be sent back to the server for encrypted computations
serialized_evaluation_keys = client.evaluation_keys.serialize()

# Initialize random weights and bias for the model
weights = np.random.randn(1, X_train.shape[1], 1)
bias = np.random.randn(1, 1, 1)

# Calculate and print the model's accuracy before any FHE-based training adjustments
initial_predictions = make_predictions(X_train, weights, bias)
initial_accuracy = calculate_accuracy(initial_predictions, y_train.flatten())
print(f"Pre-training: Accuracy = {initial_accuracy*100:.2f}%")

# Process the training data in batches for encrypted computations
batch_size = 8
num_batches = len(X_train) // batch_size

# Split the training data into manageable batches
X_train_batches = np.array_split(X_train[:num_batches * batch_size], num_batches)
y_train_batches = np.array_split(y_train[:num_batches * batch_size], num_batches)

# Iterate over each batch to perform encrypted training and accuracy evaluation
for i, (X_batch, y_batch) in enumerate(zip(X_train_batches, y_train_batches)):
    # Prepare the data for FHE computation by reshaping and quantizing
    X_batch_reshaped = X_batch.reshape(1, batch_size, -1)
    y_batch_reshaped = y_batch.reshape(1, batch_size, -1)
    quantized_inputs = model.training_quantized_module.quantize_input(X_batch_reshaped, y_batch_reshaped, weights, bias)

    # Encrypt the quantized inputs for secure server-side computation
    encrypted_inputs = client.encrypt(*quantized_inputs)
    serialized_encrypted_inputs = [encrypted_input.serialize() for encrypted_input in encrypted_inputs]

    # ----- Server-Side Computation -----
    # The server deserialize inputs, performs computation on encrypted inputs, and re-serialize the results
    deserialized_encrypted_inputs = [fhe.Value.deserialize(input) for input in serialized_encrypted_inputs]
    deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
    results = server.run(*deserialized_encrypted_inputs, evaluation_keys=deserialized_evaluation_keys)
    serialized_results = [result.serialize() for result in results]

    # ----- Client-Side Post-Computation -----
    # Decrypt and dequantize the results to update model weights and bias
    deserialized_results = [fhe.Value.deserialize(result) for result in serialized_results]
    decrypted_results = client.decrypt(*deserialized_results)
    weights, bias = model.training_quantized_module.dequantize_output(*decrypted_results)

    # Use the updated weights and bias to make predictions on the current batch
    predictions = make_predictions(X_batch, weights, bias)
    
    # Calculate and print the accuracy for the current batch
    accuracy = calculate_accuracy(predictions, y_batch.flatten())
    print(f"Iteration {i+1}/{num_batches}: Accuracy = {accuracy*100:.2f}%")