import numpy as np
import open3d as o3d


def bbox_from_points(points):
    """
    Calculates the minimum oriented bounding box for a set of 3D points using Open3D.
    
    Args:
        points: A list or an Nx3 numpy array representing 3D coordinates.
        
    Returns:
        A dictionary containing:
            'center': The center of the oriented bounding box in global coordinates.
            'extents': The dimensions (width, height, depth) of the bounding box.
            'rotation': The 3x3 rotation matrix representing the orientation.
            'corners': The 8 corners of the bounding box in global coordinates.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if len(points) == 0:
        raise ValueError("Points array cannot be empty.")
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute the minimal oriented bounding box using Open3D
    obb = pcd.get_minimal_oriented_bounding_box()
    
    x_min, y_min, z_min = -obb.extent / 2.0
    x_max, y_max, z_max = obb.extent / 2.0
    
    # 8 corners in local coordinate system, matching the previous order
    local_corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])
    
    # Transform back to global coordinates
    corners = obb.center + np.dot(local_corners, obb.R.T)
    
    return {
        'center': obb.center,
        'extents': obb.extent,
        'rotation': obb.R,
        'corners': corners
    }
