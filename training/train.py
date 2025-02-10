import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import argparse
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import io
from warnings import filterwarnings
filterwarnings('ignore')

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can import our local modules
from dataloaders.dataset import MocapDataset
from model.baseline import build_camera_matrix_model
from loss_fns.geoloss import GeometricLoss
from visualization.project_3d_2d import visualize_cameras_and_skeleton

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs, training_steps, validation_steps):
        super().__init__()
        self.num_epochs = num_epochs
        self.training_steps = training_steps
        self.validation_steps = validation_steps
        self.train_losses = []
        self.val_losses = []
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.num_epochs}")
        self.progbar = tf.keras.utils.Progbar(
            self.training_steps,
            stateful_metrics=['loss', 'avg', 'lr'],
            width=40,
            unit_name='step'
        )
        self.batch_losses = []
        
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0)
        self.batch_losses.append(loss)
        
        # Get current learning rate
        lr = float(self.model.optimizer.learning_rate.numpy())
            
        # Update progress with loss and learning rate
        self.progbar.update(
            batch + 1,
            values=[
                ('loss', loss),
                ('avg', np.mean(self.batch_losses)),
                ('lr', lr)
            ]
        )
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Calculate improvements
        if epoch > 0:
            loss_imp = (self.train_losses[-2] - train_loss) / self.train_losses[-2] * 100
            val_imp = (self.val_losses[-2] - val_loss) / self.val_losses[-2] * 100
            print(f"\nTrain: {train_loss:.4f} ({loss_imp:+.1f}%) | Val: {val_loss:.4f} ({val_imp:+.1f}%)")
        else:
            print(f"\nTrain: {train_loss:.4f} | Val: {val_loss:.4f}")

class CameraVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, dataset, freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.dataset = dataset
        self.freq = freq
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'camera_viz'))
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            for batch in self.dataset.take(1):
                inputs, targets = batch
                predictions = self.model(inputs, training=False)
                
                # Get first sample from batch
                keypoints_2d = inputs[0]  # Shape: [num_joints * 2]
                keypoints_2d = tf.reshape(keypoints_2d, [-1, 2])  # [num_joints, 2]
                
                # Convert 2D points to 3D by assuming a default depth
                num_points = tf.shape(keypoints_2d)[0]
                points_3d = tf.concat([
                    keypoints_2d,
                    tf.ones([num_points, 1]) * 2.0  # Default depth of 2 units
                ], axis=1)

                ##load 3d from dataset
                keypoints_3d = inputs[1]
                
                # Get camera matrices
                true_matrix = tf.reshape(targets[0], [4, 4])
                pred_matrix = tf.reshape(predictions[0], [4, 4])
                
                # Create figure with subplots
                fig = plt.figure(figsize=(15, 5))
                
                # 1. Original 2D keypoints
                ax1 = fig.add_subplot(131)
                ax1.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='blue', alpha=0.6)
                ax1.set_title('Original 2D Keypoints')
                ax1.set_aspect('equal')
                
                # 2. 3D visualization with cameras and skeleton
                ax2 = fig.add_subplot(132, projection='3d')
                
                visualize_cameras_and_skeleton(
                    ax2, 
                    true_matrix.numpy(),
                    pred_matrix.numpy(),
                    points_3d.numpy(),  # Pass 3D points instead of 2D
                    title='3D Scene with Cameras'
                )
                
                # 3. Reprojection comparison
                ax3 = fig.add_subplot(133)
                # Project 3D points using both cameras
                true_proj = self.project_3d_points(points_3d, true_matrix)
                pred_proj = self.project_3d_points(points_3d, pred_matrix)
                
                ax3.scatter(true_proj[:, 0], true_proj[:, 1], c='blue', alpha=0.6, label='Ground Truth')
                ax3.scatter(pred_proj[:, 0], pred_proj[:, 1], c='red', alpha=0.6, label='Predicted')
                ax3.set_title('Reprojection Comparison')
                ax3.legend()
                ax3.set_aspect('equal')
                
                plt.tight_layout()
                
                # Save to TensorBoard
                with self.file_writer.as_default():
                    tf.summary.image(f'Camera_Visualization', self.plot_to_image(fig), step=epoch)
                plt.close(fig)
    
    def project_3d_points(self, points_3d, camera_matrix):
        """Project 3D points using camera matrix"""
        # Add homogeneous coordinate
        points_h = tf.concat([points_3d, tf.ones([tf.shape(points_3d)[0], 1])], axis=-1)  # [N, 4]
        # Project
        projected = tf.matmul(points_h, tf.transpose(camera_matrix))  # [N, 4]
        # Normalize homogeneous coordinates
        return projected[:, :2] / projected[:, 2:3]
    
    def plot_to_image(self, figure):
        """Convert matplotlib figure to TensorBoard-compatible image"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

def project_points(points_3d, camera_matrix):
    # Helper function to project 3D points using camera matrix
    points_h = tf.concat([points_3d, tf.ones([tf.shape(points_3d)[0], 1])], axis=-1)
    projected = tf.matmul(points_h, tf.transpose(camera_matrix))
    return projected[:, :2] / projected[:, 2:3]

def main(data_fraction=1.0, batch_size=32, epochs=200):
    print("\n╔════ Training Configuration ═══════════════════════════════════╗")
    print(f" ║ Data Fraction: {data_fraction*100:>6.1f}%                     ║")
    print(f" ║ Batch Size:    {batch_size:>6}                                ║")
    print(f" ║ Epochs:        {epochs:>6}                                    ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    
    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' in the .env file.")

    try:
        # Load dataset with progress bar
        print("\nLoading dataset...")
        full_dataset = MocapDataset(
            uri=uri,
            db_name='ai',
            collection_name='cameraPoses',
            skeleton=None,
            data_fraction=data_fraction
        )

        # Process dataset
        def process_sample(keypoints_2d, keypoints_3d, camera_matrix, joint_names, joint_indices):
            # Ensure camera_matrix is flattened
            camera_matrix_flat = tf.reshape(camera_matrix, [-1])  # Flatten to 16 elements
            return keypoints_2d, camera_matrix_flat

        processed_dataset = full_dataset.map(process_sample)

        # Calculate sizes and split dataset
        total_size = 513253
        used_size = int(total_size * data_fraction)
        train_size = int(0.8 * used_size)
        val_size = used_size - train_size

        print(f"\nDataset split:")
        print(f"- Training samples: {train_size}")
        print(f"- Validation samples: {val_size}")

        # Split and configure datasets
        train_dataset = processed_dataset.take(train_size).cache().shuffle(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = processed_dataset.skip(train_size).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Setup model
        model = build_camera_matrix_model(
            input_dim=62,  # 31 joints * 2 coordinates
            output_dim=16,  # 4x4 matrix flattened
            dropout_rate=0.5
        )

        # Setup training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"frac{data_fraction}_batch{batch_size}_{timestamp}"
        checkpoint_dir = os.path.join(project_root, 'checkpoints', experiment_name)
        log_dir = os.path.join(project_root, 'logs', experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Setup callbacks
        callbacks = [
            TrainingProgressCallback(
                num_epochs=epochs,
                training_steps=train_size // batch_size,
                validation_steps=val_size // batch_size
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}-{val_loss:.4f}.h5'),
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch',
                profile_batch=0
            ),
            CameraVisualizationCallback(
                log_dir=log_dir,
                dataset=val_dataset
            )
        ]

        # Compile and train
        optimizer = Adam(learning_rate=0.001)
        loss_fn = GeometricLoss(alpha=0.5, beta=0.3, gamma=0.2)

        def log_loss_components(y_true, y_pred):
            # Calculate total loss using GeometricLoss
            total_loss = loss_fn(y_true, y_pred)
            
            # Log individual components with better names
            with tf.summary.record_if(True):
                # Calculate individual losses
                matrix_loss = loss_fn.frobenius_norm_loss(y_true, y_pred)
                orthogonality_loss = loss_fn.compute_orthogonality_loss(y_pred)
                scale_loss = loss_fn.compute_scale_constraint(y_pred)
                
                # Log components with descriptive names
                tf.summary.scalar('losses/matrix_difference', matrix_loss)
                tf.summary.scalar('losses/rotation_orthogonality', orthogonality_loss)
                tf.summary.scalar('losses/scale_and_translation', scale_loss)
                
                # Log tracked metrics with better organization
                metrics = loss_fn.get_metrics()
                for name, value in metrics.items():
                    tf.summary.scalar(f'metrics/{name}', value)
                
                # Log weights in their own category
                tf.summary.scalar('weights/alpha_matrix', loss_fn.alpha)
                tf.summary.scalar('weights/beta_orthogonality', loss_fn.beta)
                tf.summary.scalar('weights/gamma_scale', loss_fn.gamma)
            
            return total_loss

        model.compile(
            optimizer=optimizer,
            loss=log_loss_components,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ]
        )

        print("\n=== Starting Training ===")
        start_time = datetime.now()

        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            # Create model directory
            model_dir = os.path.join(project_root, 'weights', experiment_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model in different formats
            model_path = os.path.join(model_dir, 'model.h5')
            # Fix: ensure the weights filename ends with '.weights.h5'
            weights_path = os.path.join(model_dir, 'weights.weights.h5')
            
            # Save full model
            model.save(model_path)
            
            # Save weights separately
            model.save_weights(weights_path)
            
            print("\n╔════ Training Summary ═══════════════════════════════════╗")
            print(f" ║ Best Val Loss:  {min(history.history['val_loss']):.4f}  ║")
            print(f" ║ Final Loss:          {history.history['loss'][-1]:.4f}  ║")
            print(f" ║ Duration:                {datetime.now() - start_time}  ║")
            print("  ╚═════════════════════════════════════════════════════════╝")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            raise

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the camera matrix prediction model')
    parser.add_argument('--data-fraction', type=float, default=0.01,
                      help='Fraction of data to use (0.0 to 1.0)')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    main(
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
