import tensorflow as tf

#1. Create a random tensor of Shape (4,6)
tensor = tf.random.uniform((4,6))
print(tensor)

#2. Find its Rank and Shape using TensorFlow functions
#Rank
tensor_rank = tf.rank(tensor).numpy()
print("Rank before: ", tensor_rank)

#Shape
tensor_shape = tf.shape(tensor).numpy()
print("Shape before: ", tensor_shape)

#3. Reshape into (2,3,4) and transpose to (3,2,4)
reshaped_tensor = tf.reshape(tensor, (2,3,4))
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1,0,2])
transposed_tensor_shape = tf.shape(transposed_tensor).numpy()
transposed_tensor_rank = tf.rank(transposed_tensor).numpy()

print("Rank after: ", transposed_tensor_rank)
print("Shape after: ", transposed_tensor.shape)

#4. Broadcast a smaller tensor (1,4) to match the larger tensor and add them
small_tensor = tf.random.uniform((1,4))

final = transposed_tensor + small_tensor

print("Small tensor shape: ", small_tensor.shape)
print("Final tensor shape: ", final.shape)
print("Final tensor: ", final)