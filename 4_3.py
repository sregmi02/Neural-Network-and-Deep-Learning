
import numpy as np

# Define the scaled dot-product attention function
def scaled_dot_product_attention(Q, K, V):

    # Step 1: Compute dot product of Q and K transpose
    scores = np.dot(Q, K.T)

    # Step 2: Scale the scores by sqrt(d)
    d = K.shape[1]
    scaled_scores = scores / np.sqrt(d)

    # Step 3: Apply softmax to get attention weights
    exp_scores = np.exp(scaled_scores)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Step 4: Multiply attention weights by V
    output = np.dot(attention_weights, V)

    return attention_weights, output


# Test input matrices
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

# Call the function
attention_weights, output = scaled_dot_product_attention(Q, K, V)

# Display results
print("Attention Weights:")
print(attention_weights)

print("\nFinal Output:")
print(output)
