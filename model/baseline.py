import tensorflow as tf
from tensorflow.keras import layers, Model, constraints, initializers

# Define a Residual Block following the design in Martinez et al.
class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(
            units,
            kernel_initializer=initializers.HeNormal(),
            kernel_constraint=constraints.max_norm(1.0)
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.dense2 = layers.Dense(
            units,
            kernel_initializer=initializers.HeNormal(),
            kernel_constraint=constraints.max_norm(1.0)
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection: input added to the block output
        return x + inputs

def build_camera_matrix_model(input_dim, output_dim, dropout_rate=0.5):
    """
    Builds a feed-forward network to predict a camera matrix from 2d keypoints.
    
    Architecture:
      - Input Dense layer to expand input to 1024 dimensions
      - BatchNorm + ReLU activation
      - Two Residual Blocks (each with 2 dense layers with BN, dropout, and ReLU)
      - Final Dense layer mapping to output_dim (here, 16)
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Increase the input dimension to 1024
    x = layers.Dense(
        1024,
        kernel_initializer=initializers.HeNormal(),
        kernel_constraint=constraints.max_norm(1.0)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Apply two residual blocks
    x = ResidualBlock(1024, dropout_rate)(x)
    x = ResidualBlock(1024, dropout_rate)(x)
    
    # Final prediction layer
    outputs = layers.Dense(output_dim, kernel_initializer=initializers.HeNormal())(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="CameraMatrixPredictor")
    return model

# For testing purposes: Uncomment to build and summarize the model
# if __name__ == '__main__':
#     model = build_camera_matrix_model(input_dim=62, output_dim=16, dropout_rate=0.5)
#     model.summary()
