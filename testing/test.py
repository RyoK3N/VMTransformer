import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.baseline import build_camera_matrix_model
from dataloaders.dataset import MocapDataset
from visualization.project_3d_2d import visualize_cameras_and_skeleton

def load_model(model_path):
    """Load a trained model from the specified path."""
    try:
        # First try to load as a full model
        model = tf.keras.models.load_model(model_path, compile=False)
    except:
        # If that fails, try to load as weights
        model = build_camera_matrix_model(
            input_dim=62,  # 31 joints * 2 coordinates
            output_dim=16,  # 4x4 matrix flattened
            dropout_rate=0.0  # No dropout during inference
        )
        model.load_weights(model_path)
    
    return model

def load_test_dataset(data_fraction=0.1):
    """Load test dataset from MongoDB."""
    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' in the .env file.")

    # Create dataset
    dataset = MocapDataset(
        uri=uri,
        db_name='ai',
        collection_name='cameraPoses',
        skeleton=None,
        data_fraction=data_fraction
    )
    
    # Process dataset
    def process_sample(keypoints_2d, keypoints_3d, camera_matrix, joint_names, joint_indices):
        camera_matrix_flat = tf.reshape(camera_matrix, [-1])  # Flatten to 16 elements
        return keypoints_2d, camera_matrix_flat

    # Configure dataset
    dataset = dataset.map(process_sample).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

def compute_metrics(y_true, y_pred):
    """Compute various evaluation metrics."""
    # Reshape predictions to 4x4 matrices if they're flattened
    if y_true.shape[-1] == 16:
        y_true = y_true.reshape(-1, 4, 4)
        y_pred = y_pred.reshape(-1, 4, 4)
    
    # Calculate matrix difference (Frobenius norm)
    matrix_diff = np.mean(np.linalg.norm(y_true - y_pred, ord='fro', axis=(1, 2)))
    
    # Calculate orthogonality error for rotation part
    R_pred = y_pred[:, :3, :3]
    R_pred_t = np.transpose(R_pred, (0, 2, 1))
    I = np.eye(3)
    orthogonality = np.mean(np.linalg.norm(R_pred @ R_pred_t - I, ord='fro', axis=(1, 2)))
    
    # Calculate scale error
    scale = np.mean(np.abs(np.linalg.norm(R_pred, ord='fro', axis=(1, 2)) - np.sqrt(3)))
    
    # Calculate rotation error
    R_true = y_true[:, :3, :3]
    R_diff = np.matmul(R_true, np.transpose(R_pred, (0, 2, 1)))
    rotation_error = np.mean(np.arccos(
        np.clip((np.trace(R_diff, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    )) * 180 / np.pi
    
    # Calculate translation error
    t_true = y_true[:, :3, 3]
    t_pred = y_pred[:, :3, 3]
    translation_error = np.mean(np.linalg.norm(t_true - t_pred, axis=1))
    
    return {
        'matrix_difference': float(matrix_diff),
        'orthogonality_error': float(orthogonality),
        'scale_error': float(scale),
        'rotation_error_degrees': float(rotation_error),
        'translation_error': float(translation_error)
    }

def evaluate_model(model, test_dataset, num_visualization_samples=5):
    """Evaluate model on test dataset and generate visualizations."""
    # Collect predictions and ground truth
    all_inputs = []
    all_predictions = []
    all_targets = []
    
    print("\nGenerating predictions...")
    for batch in tqdm(test_dataset):
        inputs, targets = batch
        predictions = model(inputs, training=False)
        
        all_inputs.extend(inputs.numpy())
        all_predictions.extend(predictions.numpy())
        all_targets.extend(targets.numpy())
    
    # Convert to numpy arrays
    all_inputs = np.array(all_inputs)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_targets, all_predictions)
    
    # Generate visualizations for a few samples
    print("\nGenerating visualizations...")
    visualizations = []
    for i in range(min(num_visualization_samples, len(all_inputs))):
        fig = plt.figure(figsize=(15, 5))
        
        # Get sample data
        kp_2d = all_inputs[i].reshape(-1, 2)
        pred_matrix = all_predictions[i].reshape(4, 4)
        true_matrix = all_targets[i].reshape(4, 4)
        
        # Create 3D points
        points_3d = np.concatenate([
            kp_2d,
            np.ones((len(kp_2d), 1)) * 2.0
        ], axis=1)
        
        # 1. Original 2D keypoints
        ax1 = fig.add_subplot(131)
        ax1.scatter(kp_2d[:, 0], kp_2d[:, 1], c='blue', alpha=0.6)
        ax1.set_title('Input 2D Keypoints')
        ax1.set_aspect('equal')
        
        # 2. 3D visualization
        ax2 = fig.add_subplot(132, projection='3d')
        visualize_cameras_and_skeleton(
            ax2,
            true_matrix,
            pred_matrix,
            points_3d,
            title='3D Scene Comparison'
        )
        
        # 3. Reprojection comparison
        ax3 = fig.add_subplot(133)
        
        # Project using both matrices
        points_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        true_proj = points_h @ true_matrix.T
        true_proj = true_proj[:, :2] / true_proj[:, 2:3]
        
        pred_proj = points_h @ pred_matrix.T
        pred_proj = pred_proj[:, :2] / pred_proj[:, 2:3]
        
        ax3.scatter(true_proj[:, 0], true_proj[:, 1], c='blue', alpha=0.6, label='Ground Truth')
        ax3.scatter(pred_proj[:, 0], pred_proj[:, 1], c='red', alpha=0.6, label='Predicted')
        ax3.set_title('Reprojection Comparison')
        ax3.legend()
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        visualizations.append(fig)
    
    return metrics, visualizations

def main(model_path, output_dir, data_fraction=0.1):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = load_test_dataset(data_fraction)
    
    # Evaluate model
    metrics, visualizations = evaluate_model(model, test_dataset)
    
    # Save results
    results_dir = os.path.join(output_dir, f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save visualizations
    for i, fig in enumerate(visualizations):
        viz_path = os.path.join(results_dir, f'sample_{i+1}.png')
        fig.savefig(viz_path)
        plt.close(fig)
    print(f"Saved {len(visualizations)} visualizations to: {results_dir}")
    
    # Print metrics summary
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate camera matrix prediction model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model (.h5 file)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--data-fraction', type=float, default=0.1,
                      help='Fraction of test data to use (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        output_dir=args.output_dir,
        data_fraction=args.data_fraction
    )
