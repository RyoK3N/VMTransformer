import tensorflow as tf

class CameraMatrixLoss(tf.keras.losses.Loss):
    """
    Loss function for camera matrix prediction using Frobenius norm.
    """
    def __init__(self, alpha=0.5, name='camera_matrix_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        
    def frobenius_norm_loss(self, y_true, y_pred):
        # Reshape matrices to their 4x4 form
        y_true_matrix = tf.reshape(y_true, [-1, 4, 4])
        y_pred_matrix = tf.reshape(y_pred, [-1, 4, 4])
        
        # Compute the Frobenius norm of the difference
        diff = y_true_matrix - y_pred_matrix
        frob_norm = tf.norm(diff, ord='fro', axis=[1, 2])
        
        # Normalize by matrix size
        normalized_loss = frob_norm / tf.sqrt(tf.cast(16, tf.float32))  # 16 = 4x4
        return tf.reduce_mean(normalized_loss)
        
    def call(self, y_true, y_pred):
        return self.frobenius_norm_loss(y_true, y_pred)

class GeometricLoss(tf.keras.losses.Loss):
    """
    Advanced geometric loss that considers 3D structure and reprojection.
    Uses Frobenius norm for matrix differences.
    """
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, name='geometric_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Create metric objects for tracking
        self.matrix_loss_tracker = tf.keras.metrics.Mean(name='matrix_loss')
        self.orthogonality_loss_tracker = tf.keras.metrics.Mean(name='orthogonality_loss')
        self.scale_loss_tracker = tf.keras.metrics.Mean(name='scale_loss')
    
    def update_metrics(self, matrix_loss, orthogonality_loss, scale_loss):
        self.matrix_loss_tracker.update_state(matrix_loss)
        self.orthogonality_loss_tracker.update_state(orthogonality_loss)
        self.scale_loss_tracker.update_state(scale_loss)
    
    def get_metrics(self):
        return {
            'matrix_loss': self.matrix_loss_tracker.result(),
            'orthogonality_loss': self.orthogonality_loss_tracker.result(),
            'scale_loss': self.scale_loss_tracker.result()
        }
    
    def reset_metrics(self):
        self.matrix_loss_tracker.reset_states()
        self.orthogonality_loss_tracker.reset_states()
        self.scale_loss_tracker.reset_states()
    
    def frobenius_norm_loss(self, y_true, y_pred):
        # Reshape matrices to their 4x4 form
        y_true_matrix = tf.reshape(y_true, [-1, 4, 4])
        y_pred_matrix = tf.reshape(y_pred, [-1, 4, 4])
        
        # Compute the Frobenius norm of the difference
        diff = y_true_matrix - y_pred_matrix
        frob_norm = tf.norm(diff, ord='fro', axis=[1, 2])
        
        # Normalize by matrix size
        normalized_loss = frob_norm / tf.sqrt(tf.cast(16, tf.float32))
        return tf.reduce_mean(normalized_loss)
    
    def compute_reprojection_loss(self, points_3d, points_2d, camera_matrix):
        # Convert 3D points to homogeneous coordinates
        ones = tf.ones([tf.shape(points_3d)[0], tf.shape(points_3d)[1], 1])
        points_3d_h = tf.concat([points_3d, ones], axis=-1)
        
        # Project 3D points to 2D
        projected = tf.matmul(points_3d_h, tf.transpose(camera_matrix, [0, 2, 1]))
        projected_2d = projected[..., :2] / projected[..., 2:3]
        
        # Compute normalized L2 distance
        diff = points_2d - projected_2d
        squared_dist = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.reduce_mean(tf.sqrt(squared_dist))
    
    def compute_orthogonality_loss(self, camera_matrix):
        # Reshape to batch of 4x4 matrices if needed
        if len(camera_matrix.shape) == 2:
            camera_matrix = tf.reshape(camera_matrix, [-1, 4, 4])
            
        # Extract rotation matrix (first 3x3 block)
        rotation = camera_matrix[..., :3, :3]  # Shape: [batch_size, 3, 3]
        
        # Compute R * R^T - I (should be zero for perfect orthogonality)
        r_rt = tf.matmul(rotation, tf.transpose(rotation, perm=[0, 2, 1]))
        identity = tf.eye(3, batch_shape=[tf.shape(rotation)[0]])
        orthogonality = r_rt - identity
        
        # Use Frobenius norm for the orthogonality constraint
        frob_norm = tf.norm(orthogonality, ord='fro', axis=[1, 2])
        return tf.reduce_mean(frob_norm)
    
    def compute_scale_constraint(self, camera_matrix):
        # Reshape to batch of 4x4 matrices if needed
        if len(camera_matrix.shape) == 2:
            camera_matrix = tf.reshape(camera_matrix, [-1, 4, 4])
            
        # Extract rotation part (should have determinant = 1)
        rotation = camera_matrix[..., :3, :3]
        det = tf.linalg.det(rotation)
        scale_loss = tf.abs(det - 1.0)
        
        # Extract translation part (should be normalized)
        translation = camera_matrix[..., :3, 3]
        trans_norm = tf.norm(translation, axis=-1)
        norm_loss = tf.abs(trans_norm - 1.0)
        
        return tf.reduce_mean(scale_loss + norm_loss)
    
    def call(self, y_true, y_pred):
        # Base loss using Frobenius norm
        matrix_loss = self.frobenius_norm_loss(y_true, y_pred)
        
        # Compute individual loss components
        orthogonality_loss = self.compute_orthogonality_loss(y_pred)
        scale_loss = self.compute_scale_constraint(y_pred)
        
        # Update metrics
        self.update_metrics(matrix_loss, orthogonality_loss, scale_loss)
        
        # Compute total loss with weights
        total_loss = (
            matrix_loss + 
            self.beta * orthogonality_loss + 
            self.gamma * scale_loss
        )
        
        return total_loss
