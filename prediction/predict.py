import os
import sys
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.baseline import build_camera_matrix_model
from dataloaders.dataset import MocapDatasetGenerator

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

def predict_single(model, keypoints_2d):
    """Predict camera matrix from 2D keypoints."""
    print("\nInput keypoints shape:", keypoints_2d.shape)
    print("Number of 2D joints:", len(keypoints_2d) // 2)
    
    # Ensure keypoints are in the right shape and type
    if isinstance(keypoints_2d, tf.Tensor):
        keypoints_2d = keypoints_2d.numpy()
    
    # Add batch dimension
    input_data = np.expand_dims(keypoints_2d, axis=0)
    
    # Make prediction
    prediction = model.predict(input_data, verbose=0)
    camera_matrix = prediction[0].reshape(4, 4)
    
    return camera_matrix

def get_sample_from_mongodb(dataset, index):
    """Get a specific sample from MongoDB using index."""
    try:
        # Get specific sample using __getitem__
        print(f"\nFetching sample {index} from MongoDB...")
        sample = dataset[index]
        keypoints_2d, keypoints_3d, camera_matrix, joint_names, joint_indices = sample
        
        print("\nSample data:")
        print(f"2D keypoints shape: {keypoints_2d.shape} ({len(keypoints_2d)//2} joints)")
        print(f"3D keypoints shape: {keypoints_3d.shape} ({len(keypoints_3d)//3} joints)")
        print(f"Camera matrix shape: {camera_matrix.shape}")
        print(f"Number of joints: {len(joint_names)}")
        print("Joint names:", joint_names.numpy().tolist())
        
        # Reshape camera matrix from (16,) to (4,4)
        if isinstance(camera_matrix, tf.Tensor):
            camera_matrix = camera_matrix.numpy()
        camera_matrix = camera_matrix.reshape(4, 4)
        
        return keypoints_2d, camera_matrix
    except IndexError:
        print(f"\nError: Index {index} is out of range!")
        return None, None

def format_matrix(matrix):
    """Format a 4x4 matrix for display."""
    return "\n".join([
        " ".join([f"{x:8.3f}" for x in row])
        for row in matrix
    ])

def display_matrices(true_matrix, pred_matrix):
    """Display ground truth and predicted matrices side by side."""
    # Convert tensors to numpy if needed
    if isinstance(true_matrix, tf.Tensor):
        true_matrix = true_matrix.numpy()
    if isinstance(pred_matrix, tf.Tensor):
        pred_matrix = pred_matrix.numpy()
    
    # Ensure matrices are 4x4
    true_matrix = true_matrix.reshape(4, 4)
    pred_matrix = pred_matrix.reshape(4, 4)
    
    print("\n=== Ground Truth Matrix ===")
    print(format_matrix(true_matrix))
    
    print("\n=== Predicted Matrix ===")
    print(format_matrix(pred_matrix))
    
    # Calculate basic error metrics
    rotation_error = np.arccos(
        np.clip((np.trace(true_matrix[:3, :3] @ pred_matrix[:3, :3].T) - 1) / 2, -1.0, 1.0)
    ) * 180 / np.pi
    
    translation_error = np.linalg.norm(true_matrix[:3, 3] - pred_matrix[:3, 3])
    
    print(f"\nRotation Error: {rotation_error:.2f}°")
    print(f"Translation Error: {translation_error:.3f}")

def main(model_path):
    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' in the .env file.")

    # Load model
    print("\nLoading model...")
    model = load_model(model_path)
    
    # Create dataset generator
    dataset = MocapDatasetGenerator(
        uri=uri,
        db_name='ai',
        collection_name='cameraPoses',
        skeleton=None,
        data_fraction=1.0  # Use all data for testing
    )
    
    print("\nModel loaded and ready!")
    print(f"Available indices: 0 to {len(dataset)-1}")
    print("Enter an index number to test (or 'q' to quit)")
    
    while True:
        try:
            user_input = input("\nEnter index > ")
            
            if user_input.lower() == 'q':
                print("Exiting...")
                dataset.close_connection()  # Close MongoDB connection
                break
            
            try:
                index = int(user_input)
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
                continue
            
            # Get sample and make prediction
            keypoints_2d, true_matrix = get_sample_from_mongodb(dataset, index)
            
            if keypoints_2d is None:
                continue
                
            pred_matrix = predict_single(model, keypoints_2d)
            
            # Display results
            display_matrices(true_matrix, pred_matrix)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            dataset.close_connection()  # Close MongoDB connection
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interactive camera matrix prediction')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model (.h5 file)')
    
    args = parser.parse_args()
    main(args.model)
