import os
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from dotenv import load_dotenv
from scipy import stats
from tqdm import tqdm

class MocapDataset(tf.data.Dataset):
    """
    This class is a dataset for the GTransformer model using TensorFlow.
    It is used to load the data from the MongoDB database.
    """
    
    def __new__(cls, uri, db_name, collection_name, skeleton=None, data_fraction=1.0):
        # Pre-load all samples for the specified data fraction
        generator = MocapDatasetGenerator(
            uri=uri, 
            db_name=db_name, 
            collection_name=collection_name, 
            skeleton=skeleton,
            data_fraction=data_fraction
        )
        
        # Load samples in batches
        total_samples = len(generator)
        print(f"Pre-loading {total_samples} samples...")
        samples = []
        
        # Create progress bar
        pbar = tqdm(
            total=total_samples,
            desc="Loading samples",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # Batch loading
        BATCH_SIZE = 100
        for i in range(0, total_samples, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_samples)
            batch_samples = generator.get_batch(i, batch_end)
            samples.extend(batch_samples)
            pbar.update(len(batch_samples))
        
        pbar.close()
        
        # Convert to tensors with progress reporting
        print("\nConverting to tensors...")
        keypoints_2d = tf.stack([s[0] for s in samples])
        keypoints_3d = tf.stack([s[1] for s in samples])
        camera_matrices = tf.stack([s[2] for s in samples])
        joint_names = samples[0][3]
        joint_indices = samples[0][4]
        
        print("Dataset loaded successfully!")
        return tf.data.Dataset.from_tensor_slices((
            keypoints_2d,
            keypoints_3d,
            camera_matrices,
            tf.repeat([joint_names], total_samples, axis=0),
            tf.repeat([joint_indices], total_samples, axis=0)
        ))

class MocapDatasetGenerator:
    def __init__(self, uri, db_name, collection_name, skeleton=None, data_fraction=1.0):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._skeleton = skeleton
        self.data_fraction = data_fraction

        self.client = None
        self.collection = None
        self.joint_names = []
        self.num_joints = 0
        self._ids = []
        self.total = 0

        self._initialize_ids_and_metadata()
        if skeleton is None:
            self._infer_skeleton()

    def _initialize_ids_and_metadata(self):
        load_dotenv()
        self._connect()
        sample_doc = self.collection.find_one()
        if not sample_doc:
            raise ValueError("The collection is empty.")
        if 'kps_2d' not in sample_doc:
            raise ValueError("Documents must contain 'kps_2d' field.")

        all_joints = list(sample_doc['kps_2d'].keys())
        self.joint_names = [joint for joint in all_joints if joint.strip().lower() not in ['date', 'body']]
        self.num_joints = len(self.joint_names)

        # Get all IDs
        all_ids = list(self.collection.find({}, {'_id': 1}))
        all_ids = [doc['_id'] for doc in all_ids]
        
        # Apply data fraction
        total_available = len(all_ids)
        num_samples = int(total_available * self.data_fraction)
        
        # Randomly sample the IDs
        if self.data_fraction < 1.0:
            import random
            random.seed(42)  # For reproducibility
            self._ids = random.sample(all_ids, num_samples)
        else:
            self._ids = all_ids
            
        self.total = len(self._ids)
        print(f"Using {self.total} samples out of {total_available} ({self.data_fraction*100:.2f}%)")

    def _connect(self):
        if self.client is None:
            try:
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                self.client.server_info()
                db = self.client[self.db_name]
                self.collection = db[self.collection_name]
            except Exception as e:
                raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    def parse_keypoints(self, keypoints_dict):
        keypoints_flat = []
        for joint in self.joint_names:
            coords = keypoints_dict.get(joint, [0.0, 0.0])
            if len(coords) < 2:
                coords = [0.0, 0.0]
            keypoints_flat.extend(coords[:2])
        
        keypoints_array = np.array(keypoints_flat, dtype=np.float32)
        return self.set_precision(data=keypoints_array)

    def parse_keypoints_3d(self, keypoints_dict):
        keypoints_flat = []
        for joint in self.joint_names:
            coords = keypoints_dict.get(joint, [0.0, 0.0, 0.0])
            if len(coords) < 3:
                coords = [0.0, 0.0, 0.0]
            keypoints_flat.extend(coords[:3])
        
        keypoints_array = np.array(keypoints_flat, dtype=np.float32)
        return self.set_precision(data=keypoints_array)

    def set_precision(self, data=None, precision=4):
        if data is not None:
            return np.round(data, decimals=precision)
        
        if hasattr(self, 'camera_matrix'):
            self.cameras = np.round(self.cameras, decimals=precision)
        if hasattr(self, 'keypoints'):
            self.keypoints = np.round(self.keypoints, decimals=precision)

    def clip_outliers_camera_matrix(self, camera_matrix):
        camera_matrix = camera_matrix.reshape(4, 4)
        scaled_camera_matrix = camera_matrix.copy()
        translation = camera_matrix[3, :3]
        min_val = translation.min()
        max_val = translation.max()
        scaled_translation = (translation - min_val) / (max_val - min_val)
        scaled_camera_matrix[3, :3] = scaled_translation
        scaled_camera_matrix = scaled_camera_matrix.flatten()
        return scaled_camera_matrix

    def parse_camera_matrix(self, camera_matrix_list):
        if len(camera_matrix_list) != 16:
            raise ValueError("Camera matrix must have 16 elements.")
        
        camera_matrix = np.array(camera_matrix_list, dtype=np.float32)
        camera_matrix = self.set_precision(data=camera_matrix)
        return self.clip_outliers_camera_matrix(camera_matrix)

    def __len__(self):
        return self.total

    def get_batch(self, start_idx, end_idx):
        """Load multiple samples at once"""
        if start_idx < 0 or end_idx > self.total:
            raise IndexError(f"Index range [{start_idx}, {end_idx}) is out of bounds")
        
        self._connect()
        batch_ids = self._ids[start_idx:end_idx]
        
        # Fetch multiple documents at once
        documents = list(self.collection.find({'_id': {'$in': batch_ids}}))
        
        # Process all documents
        batch_samples = []
        for doc in documents:
            keypoints_dict = doc.get('kps_2d', {})
            keypoints_flattened = self.parse_keypoints(keypoints_dict)

            keypoints_dict_3d = doc.get('kps_3d', {})
            keypoints_flattened_3d = self.parse_keypoints_3d(keypoints_dict_3d)

            camera_matrix_list = doc.get('camera_matrix', [])
            camera_matrix = self.parse_camera_matrix(camera_matrix_list)

            # Convert to TensorFlow tensors
            keypoints_tensor = tf.convert_to_tensor(keypoints_flattened, dtype=tf.float32)
            keypoints_tensor_3d = tf.convert_to_tensor(keypoints_flattened_3d, dtype=tf.float32)
            camera_matrix_tensor = tf.convert_to_tensor(camera_matrix, dtype=tf.float32)
            joint_names_tensor = tf.convert_to_tensor(self.joint_names)
            joint_indices_tensor = tf.range(len(self.joint_names), dtype=tf.int32)

            batch_samples.append((
                keypoints_tensor, 
                keypoints_tensor_3d, 
                camera_matrix_tensor,
                joint_names_tensor, 
                joint_indices_tensor
            ))
            
        return batch_samples

    def __getitem__(self, idx):
        """Keep this for compatibility but use get_batch for actual loading"""
        return self.get_batch(idx, idx + 1)[0]

    def close_connection(self):
        if self.client:
            self.client.close()
            self.client = None

    def _infer_skeleton(self):
        """Infer skeleton structure from the data"""
        print("Inferring skeleton structure from data...")
        
        # Get a sample document
        sample_doc = self.collection.find_one()
        
        # Get all joints from 2D and 3D data
        joints_2d = set(sample_doc.get('kps_2d', {}).keys())
        joints_3d = set(sample_doc.get('kps_3d', {}).keys())
        
        # Remove any non-joint fields
        exclude = {'date', 'body', '_id'}
        joints = sorted(list((joints_2d | joints_3d) - exclude))
        
        # Infer left/right joints
        joints_left = sorted([j for j in joints if j.lower().startswith('left')])
        joints_right = sorted([j for j in joints if j.lower().startswith('right')])
        
        # Infer connections based on common naming patterns
        connections = []
        common_parents = ['Head', 'Neck', 'Chest', 'Spine', 'Hips']
        
        for joint in joints:
            # Connect to parent joint if it exists
            for parent in common_parents:
                if parent in joint and parent != joint:
                    connections.append((parent, joint))
            
            # Connect left/right pairs
            if joint.startswith('Left'):
                right_joint = 'Right' + joint[4:]
                if right_joint in joints:
                    connections.append((joint, right_joint))

        self._skeleton = {
            'joints': joints,
            'connections': connections,
            'joints_left': joints_left,
            'joints_right': joints_right
        }
        
        print(f"Inferred {len(joints)} joints and {len(connections)} connections")
        return self._skeleton
 