import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox
import os
from dotenv import load_dotenv
from pytorch_dataset import MocapDataset


class Camera:
    def __init__(self, position=np.array([2, -4, 2]), up=np.array([0, 0, 1]), size=0.3):
        self._position = position
        self._up = up
        self._size = size
        self._direction = None
        self._right = None
        self._vertices = None
        
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = np.array(value)
        self._update_camera()
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
        self._update_camera()
    
    def look_at(self, target):
        self._direction = target - self.position
        self._direction = self._direction / np.linalg.norm(self._direction)
        self._update_camera()
    
    def _update_camera(self):
        if self._direction is None:
            return
            
        self._right = np.cross(self._direction, self._up)
        self._right = self._right / np.linalg.norm(self._right)
        self._up = np.cross(self._right, self._direction)
        self._up = self._up / np.linalg.norm(self._up)
        
        self._update_vertices()
    
    def _update_vertices(self):
        self._vertices = np.array([
            self.position,
            self.position + self.size * (-self._right + self._direction - self._up),
            self.position + self.size * (self._right + self._direction - self._up),
            self.position + self.size * (self._right + self._direction + self._up),
            self.position + self.size * (-self._right + self._direction + self._up)
        ])


class KeyPoint:
    def __init__(self, dataset_item, connections):
        self.kps2d = dataset_item[0].detach().numpy().reshape(-1, 2)
        self.kps3d = dataset_item[1].detach().numpy().reshape(-1, 3)
        self.cmx = dataset_item[2].detach().numpy().reshape(4, 4)  
        self.joint_names = dataset_item[3]
        self.joint_indices = dataset_item[4]
        self.connections = connections 
        self.connection_indices = [(self.joint_names.index(x), self.joint_names.index(y)) 
                                 for x,y in self.connections 
                                 if x in self.joint_names and y in self.joint_names]
        self._vertices = None
        self._edges = None
        self._update_geometry()
    
    @property
    def center_3d(self):
        center_x = self.kps3d[:,0].mean()
        center_y = self.kps3d[:,1].mean()
        center_z = self.kps3d[:,2].mean()
        return np.array([center_x, center_y, center_z])
    
    @property
    def center_2d(self):
        center_x = self.kps2d[:,0].mean()
        center_y = self.kps2d[:,1].mean()
        return np.array([center_x, center_y])
    
    @property
    def scale(self):
        max_val = np.max(self.kps3d)
        min_val = np.min(self.kps3d)
        return (max_val, min_val)
    
    def _update_geometry(self):
        self.origin = self.center_3d
        self.size = self.scale

        self._vertices = self.kps3d
        
        self._edges = []
        for start_idx, end_idx in self.connection_indices:
            self._edges.append([self._vertices[start_idx], self._vertices[end_idx]])

    def get_camera_params(self):
        # Extract rotation matrix (3x3) and translation vector from view matrix
        rotation = self.cmx[:3, :3]
        translation = self.cmx[:3, 3]
        
        # Camera position is -R^T * t
        camera_position = -np.linalg.inv(rotation) @ translation
        
        # Camera up vector (third row of rotation matrix)
        up_vector = rotation[2, :]
        
        # Camera direction (negative of third column of rotation matrix)
        direction = -rotation[:, 2]
        
        return camera_position, up_vector, direction


class Scene3DViewer:
    def __init__(self, camera, dataset, connections):
        self.dataset = dataset
        self.connections = connections
        
        # Create figure with proper spacing for controls
        self.fig = plt.figure(figsize=(24, 12))
        
        # Create proper spacing for subplots and controls
        self.fig.subplots_adjust(
            left=0.05,    # Left margin
            right=0.95,   # Right margin
            bottom=0.3,   # Bottom margin (increased for controls)
            top=0.95,     # Top margin
            wspace=0.3,   # Width space between subplots
            hspace=0.2    # Height space between subplots
        )
        
        # Create subplots with proper spacing
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        self.ax = self.fig.add_subplot(gs[0], projection='3d')  # 3D view
        self.ax2 = self.fig.add_subplot(gs[1])  # Reference camera view
        self.ax3 = self.fig.add_subplot(gs[2])  # Dataset camera view
        
        self.camera = camera
        self.cmx_camera = None  # Camera for CMX visualization
        self.kps = None
        
        # Camera intrinsics (can be adjusted)
        self.camera_intrinsics = np.array([
            [800., 0., 320.],
            [0., 800., 240.],
            [0., 0., 1.]
        ])
        
        self.setup_sliders()
        self.setup_frame_input()
        
    def setup_sliders(self):
        slider_color = 'lightgoldenrodyellow'
        slider_height = 0.02
        slider_width = 0.15
        
        # Calculate vertical positions for sliders
        y_positions = {
            'top': 0.22,      # Top row
            'middle': 0.17,   # Middle row
            'bottom': 0.12,   # Bottom row
            'translation': 0.07,  # Translation controls
            'view': 0.02      # View controls
        }
        
        # Calculate horizontal positions for slider columns
        x_positions = {
            'left': 0.08,     # Left column
            'mid_left': 0.32, # Middle-left column
            'mid_right': 0.56,# Middle-right column
            'right': 0.80     # Right column
        }
        
        # Reference camera position sliders (left column)
        ax_x = plt.axes([x_positions['left'], y_positions['top'], slider_width, slider_height], facecolor=slider_color)
        ax_y = plt.axes([x_positions['left'], y_positions['middle'], slider_width, slider_height], facecolor=slider_color)
        ax_z = plt.axes([x_positions['left'], y_positions['bottom'], slider_width, slider_height], facecolor=slider_color)
        
        self.slider_x = Slider(ax_x, 'Ref Cam X', -100.0, 100.0, valinit=self.camera.position[0])
        self.slider_y = Slider(ax_y, 'Ref Cam Y', -100.0, 100.0, valinit=self.camera.position[1])
        self.slider_z = Slider(ax_z, 'Ref Cam Z', -100.0, 100.0, valinit=self.camera.position[2])
        
        # Dataset camera matrix sliders with increased ranges
        ax_r11 = plt.axes([x_positions['mid_left'], y_positions['top'], slider_width, slider_height], facecolor=slider_color)
        ax_r12 = plt.axes([x_positions['mid_left'], y_positions['middle'], slider_width, slider_height], facecolor=slider_color)
        ax_r13 = plt.axes([x_positions['mid_left'], y_positions['bottom'], slider_width, slider_height], facecolor=slider_color)
        ax_t1 = plt.axes([x_positions['mid_left'], y_positions['translation'], slider_width, slider_height], facecolor=slider_color)
        
        ax_r21 = plt.axes([x_positions['mid_right'], y_positions['top'], slider_width, slider_height], facecolor=slider_color)
        ax_r22 = plt.axes([x_positions['mid_right'], y_positions['middle'], slider_width, slider_height], facecolor=slider_color)
        ax_r23 = plt.axes([x_positions['mid_right'], y_positions['bottom'], slider_width, slider_height], facecolor=slider_color)
        ax_t2 = plt.axes([x_positions['mid_right'], y_positions['translation'], slider_width, slider_height], facecolor=slider_color)
        
        ax_r31 = plt.axes([x_positions['right'], y_positions['top'], slider_width, slider_height], facecolor=slider_color)
        ax_r32 = plt.axes([x_positions['right'], y_positions['middle'], slider_width, slider_height], facecolor=slider_color)
        ax_r33 = plt.axes([x_positions['right'], y_positions['bottom'], slider_width, slider_height], facecolor=slider_color)
        ax_t3 = plt.axes([x_positions['right'], y_positions['translation'], slider_width, slider_height], facecolor=slider_color)
        
        # Initialize camera matrix sliders
        self.matrix_sliders = {
            'r11': Slider(ax_r11, 'R11', -2.0, 2.0, valinit=1.0),
            'r12': Slider(ax_r12, 'R12', -2.0, 2.0, valinit=0.0),
            'r13': Slider(ax_r13, 'R13', -2.0, 2.0, valinit=0.0),
            'r21': Slider(ax_r21, 'R21', -2.0, 2.0, valinit=0.0),
            'r22': Slider(ax_r22, 'R22', -2.0, 2.0, valinit=1.0),
            'r23': Slider(ax_r23, 'R23', -2.0, 2.0, valinit=0.0),
            'r31': Slider(ax_r31, 'R31', -2.0, 2.0, valinit=0.0),
            'r32': Slider(ax_r32, 'R32', -2.0, 2.0, valinit=0.0),
            'r33': Slider(ax_r33, 'R33', -2.0, 2.0, valinit=1.0),
            't1': Slider(ax_t1, 'T1', -50.0, 50.0, valinit=0.0),
            't2': Slider(ax_t2, 'T2', -50.0, 50.0, valinit=0.0),
            't3': Slider(ax_t3, 'T3', -50.0, 50.0, valinit=0.0),
        }
        
        # View angle sliders
        ax_elev = plt.axes([x_positions['left'], y_positions['view'], slider_width, slider_height], facecolor=slider_color)
        ax_azim = plt.axes([x_positions['mid_left'], y_positions['view'], slider_width, slider_height], facecolor=slider_color)
        
        self.slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=20)
        self.slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=45)

        # Add labels for control sections
        plt.figtext(0.08, 0.25, 'Reference Camera Controls:', fontsize=10, fontweight='bold')
        plt.figtext(0.32, 0.25, 'Dataset Camera Matrix Controls:', fontsize=10, fontweight='bold')
        plt.figtext(0.08, 0.04, 'View Controls:', fontsize=10, fontweight='bold')

        def update(val):
            if self.kps is not None:
                # Update reference (green) camera position from position sliders
                ref_position = np.array([
                    self.slider_x.val,
                    self.slider_y.val,
                    self.slider_z.val
                ])
                self.camera.position = ref_position
                self.camera.look_at(self.kps.center_3d)
                
                # Update dataset (yellow) camera from matrix sliders
                matrix = np.eye(4)
                # Build rotation matrix
                matrix[:3, :3] = np.array([
                    [self.matrix_sliders['r11'].val, self.matrix_sliders['r12'].val, self.matrix_sliders['r13'].val],
                    [self.matrix_sliders['r21'].val, self.matrix_sliders['r22'].val, self.matrix_sliders['r23'].val],
                    [self.matrix_sliders['r31'].val, self.matrix_sliders['r32'].val, self.matrix_sliders['r33'].val]
                ])
                # Add translation
                matrix[:3, 3] = np.array([
                    self.matrix_sliders['t1'].val,
                    self.matrix_sliders['t2'].val,
                    self.matrix_sliders['t3'].val
                ])
                
                # Ensure rotation matrix is orthogonal
                U, _, Vt = np.linalg.svd(matrix[:3, :3])
                matrix[:3, :3] = U @ Vt
                
                # Update dataset camera
                if self.cmx_camera is not None:
                    position, up, direction = self.get_camera_params_from_matrix(matrix)
                    self.cmx_camera.position = position
                    self.cmx_camera._up = up
                    self.cmx_camera._direction = direction
                    self.cmx_camera._update_camera()
                    # Store the current matrix for reprojection
                    self.kps.cmx = matrix
                
            self.ax.view_init(elev=self.slider_elev.val, azim=self.slider_azim.val)
            self.setup_plot()
            self.fig.canvas.draw_idle()
        
        # Register update function for all sliders
        for slider in [self.slider_x, self.slider_y, self.slider_z,
                      self.slider_elev, self.slider_azim, *self.matrix_sliders.values()]:
            slider.on_changed(update)
    
    def get_camera_params_from_matrix(self, matrix):
        """Extract camera parameters from transformation matrix."""
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        
        # Ensure rotation matrix is orthogonal
        U, _, Vt = np.linalg.svd(rotation)
        rotation = U @ Vt
        
        # Camera position is -R^T * t
        position = -rotation.T @ translation
        
        # Camera direction (negative of third column of rotation)
        direction = -rotation[:, 2]
        
        # Camera up vector (third row of rotation)
        up = rotation[2, :]
        
        return position, up, direction

    def project_points(self, points_3d, camera_matrix, camera_intrinsics):
        """Project 3D points using camera matrix and intrinsics."""
        # Convert to homogeneous coordinates
        points_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        # Apply camera extrinsics
        points_cam = points_h @ camera_matrix.T
        
        # Perspective division
        points_2d = points_cam[:, :3] / points_cam[:, 2:3]
        
        # Apply camera intrinsics
        points_pixel = points_2d @ camera_intrinsics.T
        
        return points_pixel[:, :2]  # Return only x, y coordinates
    
    def setup_frame_input(self):
        ax_frame = plt.axes([0.75, 0.03, 0.1, 0.02])
        self.text_frame = TextBox(ax_frame, 'Frame', initial='0')
        
        def update_frame(text):
            try:
                frame_idx = int(text)
                if frame_idx >= 0:
                    dataset_item = self.dataset.__getitem__(frame_idx)
                    self.kps = KeyPoint(dataset_item, self.connections)
                    
                    # Get matrix from dataset
                    matrix = self.kps.cmx
                    
                    # Update dataset camera (yellow)
                    position, up, direction = self.get_camera_params_from_matrix(matrix)
                    if self.cmx_camera is None:
                        self.cmx_camera = Camera(position=position, up=up, size=self.camera.size)
                        self.cmx_camera._direction = direction
                        self.cmx_camera._update_camera()
                    else:
                        self.cmx_camera.position = position
                        self.cmx_camera._up = up
                        self.cmx_camera._direction = direction
                        self.cmx_camera._update_camera()
                    
                    # Update matrix sliders
                    self.matrix_sliders['r11'].set_val(matrix[0, 0])
                    self.matrix_sliders['r12'].set_val(matrix[0, 1])
                    self.matrix_sliders['r13'].set_val(matrix[0, 2])
                    self.matrix_sliders['r21'].set_val(matrix[1, 0])
                    self.matrix_sliders['r22'].set_val(matrix[1, 1])
                    self.matrix_sliders['r23'].set_val(matrix[1, 2])
                    self.matrix_sliders['r31'].set_val(matrix[2, 0])
                    self.matrix_sliders['r32'].set_val(matrix[2, 1])
                    self.matrix_sliders['r33'].set_val(matrix[2, 2])
                    self.matrix_sliders['t1'].set_val(matrix[0, 3])
                    self.matrix_sliders['t2'].set_val(matrix[1, 3])
                    self.matrix_sliders['t3'].set_val(matrix[2, 3])
                    
                    # Update reference camera (green)
                    self.camera.look_at(self.kps.center_3d)
                    
                    self.setup_plot()
                    self.fig.canvas.draw_idle()
            except ValueError:
                pass
        
        self.text_frame.on_submit(update_frame)
    
    def setup_plot(self):
        self.ax.cla()
        self.ax2.cla()
        self.ax3.cla()
        
        if self.kps is not None:
            # Draw 3D scene
            self.draw_keypoints()
            self.draw_camera()
            self.draw_cmx_camera()
            
            # Calculate and display reprojection error
            error_stats = self.calculate_reprojection_error()
            if error_stats:
                error_text = f"Mean Reprojection Error: {error_stats['mean_error']:.2f} pixels\n"
                error_text += f"Max Error: {error_stats['max_error']:.2f}, Min Error: {error_stats['min_error']:.2f}"
                plt.figtext(0.5, 0.97, error_text, ha='center', va='top')
            
            # Set 3D plot properties
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Scene')
            
            max_val = np.max(np.abs(self.kps.kps3d))
            scale = max_val * 1.5
            self.ax.set_xlim(-scale, scale)
            self.ax.set_ylim(-scale, scale)
            self.ax.set_zlim(-scale, scale)
            
            # Draw reference camera view
            self.draw_camera_view(self.ax2, self.camera, "Reference Camera View")
            
            # Draw dataset camera view
            if self.cmx_camera is not None:
                self.draw_camera_view(self.ax3, self.cmx_camera, "Dataset Camera View")
    
    def draw_camera(self):
        vertices = self.camera._vertices
        if vertices is not None:

            for i in range(1, 5):
                self.ax.plot([vertices[0][0], vertices[i][0]],
                           [vertices[0][1], vertices[i][1]],
                           [vertices[0][2], vertices[i][2]],
                           'g-', linewidth=2)
            
            # Draw base
            base_indices = [1, 2, 3, 4, 1]
            x = [vertices[i][0] for i in base_indices]
            y = [vertices[i][1] for i in base_indices]
            z = [vertices[i][2] for i in base_indices]
            self.ax.plot(x, y, z, 'g-', linewidth=2)
            
            # Draw camera position
            self.ax.scatter([vertices[0][0]], [vertices[0][1]], [vertices[0][2]],
                          color='g', s=100)
    
    def draw_keypoints(self):
        vertices = self.kps._vertices
        edges = self.kps._edges
        
        # Draw vertices
        self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       color='b', s=50)
        
        # Draw edges
        for edge in edges:
            start, end = edge
            self.ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        'r-', linewidth=2)
    
    def draw_camera_view(self, ax, camera, title):
        """Draw projected view for a specific camera."""
        vertices = self.kps._vertices
        edges = self.kps._edges
        
        if camera == self.cmx_camera:
            # For dataset camera, calculate reprojection matrix from 2D-3D correspondences
            camera_matrix = self.calculate_reprojection_matrix(self.kps.kps2d, vertices)
            projected_points = self.project_points(vertices, camera_matrix, self.camera_intrinsics)
            error = np.mean(np.linalg.norm(self.kps.kps2d - projected_points, axis=1))
            title = f"{title}\nReprojection Error: {error:.2f} pixels"
        else:
            # For reference camera, calculate projection
            camera_right = camera._right
            camera_up = camera._up
            camera_dir = camera._direction
            rotation_matrix = np.column_stack([camera_right, np.cross(camera_up, camera_right), camera_up])
            translation = -rotation_matrix @ camera.position
            
            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = rotation_matrix
            camera_matrix[:3, 3] = translation
            
            projected_points = self.project_points(vertices, camera_matrix, self.camera_intrinsics)
        
        # Draw projected points and edges
        ax.scatter(projected_points[:, 0], projected_points[:, 1], c='b', s=30)
        
        # Draw edges
        for edge in edges:
            start_idx = np.where(np.all(vertices == edge[0], axis=1))[0][0]
            end_idx = np.where(np.all(vertices == edge[1], axis=1))[0][0]
            ax.plot([projected_points[start_idx, 0], projected_points[end_idx, 0]],
                   [projected_points[start_idx, 1], projected_points[end_idx, 1]], 'r-')
        
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

    def draw_cmx_camera(self):
        if self.kps is None:
            return
            
        # Get camera parameters from the view matrix
        position, up, direction = self.kps.get_camera_params()
        
        # Create a temporary camera object for visualization
        if self.cmx_camera is None:
            self.cmx_camera = Camera(position=position, up=up, size=self.camera.size)
            self.cmx_camera._direction = direction
            self.cmx_camera._update_camera()
        
        vertices = self.cmx_camera._vertices
        if vertices is not None:
            for i in range(1, 5):
                self.ax.plot([vertices[0][0], vertices[i][0]],
                           [vertices[0][1], vertices[i][1]],
                           [vertices[0][2], vertices[i][2]],
                           'y-', linewidth=2)  # yellow for CMX camera
            
            # Draw base
            base_indices = [1, 2, 3, 4, 1]
            x = [vertices[i][0] for i in base_indices]
            y = [vertices[i][1] for i in base_indices]
            z = [vertices[i][2] for i in base_indices]
            self.ax.plot(x, y, z, 'y-', linewidth=2)
            
            # Draw camera position
            self.ax.scatter([vertices[0][0]], [vertices[0][1]], [vertices[0][2]],
                          color='y', s=100)  # yellow for CMX camera

    def calculate_reprojection_error(self):
        """Calculate reprojection error between dataset 2D points and projected 3D points."""
        if self.kps is None:
            return None
            
        # Get actual 2D keypoints from dataset
        actual_2d = self.kps.kps2d
        
        # Project 3D points using dataset camera matrix
        camera_matrix = self.kps.cmx
        projected_2d = self.project_points(self.kps._vertices, camera_matrix, self.camera_intrinsics)
        
        # Calculate error for each point
        errors = np.linalg.norm(actual_2d - projected_2d, axis=1)
        
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'std_error': np.std(errors),
            'errors': errors
        }

    def calculate_reprojection_matrix(self, points_2d, points_3d):
        """Calculate reprojection matrix from 2D-3D point correspondences."""
        # Add homogeneous coordinate to 3D points
        points_3d_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        # Solve for camera matrix using least squares
        # P * X = x, where P is the camera matrix we want to find
        A = []
        b = []
        for i in range(len(points_2d)):
            x, y = points_2d[i]
            X = points_3d_h[i]
            A.append([X[0], X[1], X[2], X[3], 0, 0, 0, 0, -x*X[0], -x*X[1], -x*X[2], -x*X[3]])
            A.append([0, 0, 0, 0, X[0], X[1], X[2], X[3], -y*X[0], -y*X[1], -y*X[2], -y*X[3]])
            b.extend([0, 0])
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve using SVD for better numerical stability
        U, S, Vh = np.linalg.svd(A)
        P = Vh[-1].reshape(3, 4)
        
        # Convert to 4x4 matrix
        camera_matrix = np.eye(4)
        camera_matrix[:3, :] = P
        
        return camera_matrix


def main():
    # Load environment variables
    load_dotenv()
    
    uri = os.getenv('URI')
    db_name = os.getenv('MONGO_DB_NAME')
    collection_name = os.getenv('MONGO_COLLECTION_NAME')

    dataset = MocapDataset(uri, db_name, collection_name, skeleton=None)
    
    # Define connections based on your skeleton structure
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
    
    # Create initial keypoints
    dataset_item = dataset.__getitem__(0)
    kps = KeyPoint(dataset_item, connections)
    max_val = np.max(np.abs(kps.kps3d))
    
    # Initialize camera with position based on the skeleton scale
    camera = Camera(
        position=np.array([max_val, -max_val*2, max_val]),
        size=max_val/10  # Adjust camera size relative to skeleton
    )
    
    viewer = Scene3DViewer(camera, dataset, connections)
    viewer.kps = kps
    viewer.camera.look_at(viewer.kps.center_3d)
    plt.show()



if __name__ == '__main__':
    main()