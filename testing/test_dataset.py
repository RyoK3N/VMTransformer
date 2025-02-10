import pytest
import os
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from dataloaders.dataset import MocapDataset, MocapDatasetGenerator

@pytest.fixture
def mock_skeleton():
    connections = [
        ('Head', 'Neck'), ('Neck', 'Chest'),
        ('Chest',"LeftShoulder"),('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForearm'), ('LeftForearm', 'LeftHand'),
        ('Chest', 'RightShoulder'), ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForearm'), ('RightForearm', 'RightHand'),
        ('Hips', 'LeftThigh'), ('LeftThigh', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'), ('Hips', 'RightThigh'),
        ('RightThigh', 'RightLeg'), ('RightLeg', 'RightFoot'),
        ('RightHand', 'RightFinger'), ('RightFinger', 'RightFingerEnd'),
        ('LeftHand', 'LeftFinger'), ('LeftFinger', 'LeftFingerEnd'),
        ('Head', 'HeadEnd'), ('RightFoot', 'RightHeel'),
        ('RightHeel', 'RightToe'), ('RightToe', 'RightToeEnd'),
        ('LeftFoot', 'LeftHeel'), ('LeftHeel', 'LeftToe'),
        ('LeftToe', 'LeftToeEnd'),
        ('SpineLow', 'Hips'), ('SpineMid', 'SpineLow'), ('Chest', 'SpineMid')
    ]
    
    joints_left = [
        'LeftShoulder', 'LeftArm', 'LeftForearm', 'LeftHand', 'LeftFinger', 'LeftFingerEnd',
        'LeftThigh', 'LeftLeg', 'LeftFoot', 'LeftHeel', 'LeftToe', 'LeftToeEnd'
    ]

    joints_right = [
        'RightShoulder', 'RightArm', 'RightForearm', 'RightHand', 'RightFinger', 'RightFingerEnd',
        'RightThigh', 'RightLeg', 'RightFoot', 'RightHeel', 'RightToe', 'RightToeEnd'
    ]
    
    return {
        'connections': connections,
        'joints_left': joints_left,
        'joints_right': joints_right
    }

@pytest.fixture
def dataset_generator(mock_skeleton):
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' in the .env file.")
    
    dataset = MocapDatasetGenerator(
        uri=uri,
        db_name='ai',
        collection_name='cameraPoses',
        skeleton=mock_skeleton
    )
    yield dataset
    dataset.close_connection()

@pytest.fixture
def tf_dataset(mock_skeleton):
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' in the .env file.")
    
    return MocapDataset(
        uri=uri,
        db_name='ai',
        collection_name='cameraPoses',
        skeleton=mock_skeleton
    )

def test_dataset_initialization(dataset_generator):
    """Testing if the dataset is properly initialized"""
    assert dataset_generator is not None
    assert dataset_generator.client is not None
    assert dataset_generator.collection is not None
    assert len(dataset_generator.joint_names) > 0
    assert dataset_generator.num_joints > 0
    assert dataset_generator.total > 0

def test_dataset_getitem(dataset_generator):
    """Testing if the __getitem__ returns correct data structures"""
    idx = 0
    keypoints_2d, keypoints_3d, camera_matrix, joint_names, joint_indices = dataset_generator[idx]
    
    assert isinstance(keypoints_2d, tf.Tensor)
    assert isinstance(keypoints_3d, tf.Tensor)
    assert isinstance(camera_matrix, tf.Tensor)
    assert isinstance(joint_names, tf.Tensor)
    assert isinstance(joint_indices, tf.Tensor)
    
    assert keypoints_2d.shape[0] == dataset_generator.num_joints * 2  # x,y coordinates
    assert keypoints_3d.shape[0] == dataset_generator.num_joints * 3  # x,y,z coordinates
    assert camera_matrix.shape[0] == 16  # 4x4 matrix flattened

def test_parse_keypoints(dataset_generator):
    """Testing the keypoint parsing"""
    sample_keypoints = {
        'Head': [1.0, 2.0],
        'Neck': [3.0, 4.0],
        'Chest': [5.0, 6.0]
    }
    
    parsed = dataset_generator.parse_keypoints(sample_keypoints)
    assert isinstance(parsed, np.ndarray)
    assert parsed.dtype == np.float32
    
def test_parse_camera_matrix(dataset_generator):
    """Testing the camera matrix parsing and scaling"""
    sample_matrix = list(range(16))  # 0-15 as a 1D list
    parsed = dataset_generator.parse_camera_matrix(sample_matrix)
    
    assert isinstance(parsed, np.ndarray)
    assert parsed.dtype == np.float32
    assert len(parsed) == 16
    
    matrix_reshaped = parsed.reshape(4, 4)
    translation = matrix_reshaped[3, :3]
    assert np.all((translation >= 0) & (translation <= 1))

def test_dataset_length(dataset_generator):
    """Test if dataset length is correct"""
    assert len(dataset_generator) == dataset_generator.total
    assert dataset_generator.total > 0

def test_invalid_index(dataset_generator):
    """Test if accessing invalid index raises error"""
    with pytest.raises(IndexError):
        _ = dataset_generator[-1]
    with pytest.raises(IndexError):
        _ = dataset_generator[len(dataset_generator)]

def test_tf_dataset(tf_dataset):
    """Test TensorFlow dataset functionality"""
    # Test if we can iterate over the dataset
    for data in tf_dataset.take(1):
        keypoints_2d, keypoints_3d, camera_matrix, joint_names, joint_indices = data
        assert isinstance(keypoints_2d, tf.Tensor)
        assert isinstance(keypoints_3d, tf.Tensor)
        assert isinstance(camera_matrix, tf.Tensor)
        assert isinstance(joint_names, tf.Tensor)
        assert isinstance(joint_indices, tf.Tensor)
        break

if __name__ == '__main__':
    pytest.main([__file__])
