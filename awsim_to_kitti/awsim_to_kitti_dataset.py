import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from autoware_perception_msgs.msg import DetectedObjects
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import math

LEFT_TOPIC = '/sensing/camera_left/traffic_light/image_raw'
RIGHT_TOPIC = '/sensing/camera_right/traffic_light/image_raw'
LIDAR_TOPIC = '/sensing/lidar/top/pointcloud_raw'
DETECTIONS_TOPIC = '/perception/object_recognition/detection/centerpoint/objects'

DATASET_DIR = '/home/arka/ros2_ws/src/awsim_to_kitti/awsim_test_dataset_dummy'
IMG_LEFT_DIR = os.path.join(DATASET_DIR, 'image_2')
IMG_RIGHT_DIR = os.path.join(DATASET_DIR, 'image_3')
LIDAR_DIR = os.path.join(DATASET_DIR, 'velodyne')
LABEL_DIR = os.path.join(DATASET_DIR, 'label_2')

for d in [IMG_LEFT_DIR, IMG_RIGHT_DIR, LIDAR_DIR, LABEL_DIR]:
    os.makedirs(d, exist_ok=True)

class AwsimToKittiDataset(Node):
    def __init__(self):
        super().__init__('awsim_to_kitti_dataset')
        self.bridge = CvBridge()
        self.counter = 0

        self.left_sub = Subscriber(self, Image, LEFT_TOPIC, qos_profile=qos_profile_sensor_data)
        self.right_sub = Subscriber(self, Image, RIGHT_TOPIC, qos_profile=qos_profile_sensor_data)
        self.lidar_sub = Subscriber(self, PointCloud2, LIDAR_TOPIC, qos_profile=qos_profile_sensor_data)
        self.det_sub = Subscriber(self, DetectedObjects, DETECTIONS_TOPIC, qos_profile=qos_profile_sensor_data)

        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.lidar_sub, self.det_sub],
            queue_size=10, slop=0.05)
        #self.ts = ApproximateTimeSynchronizer(
         #   [self.left_sub, self.right_sub, self.det_sub],
         #   queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
    
    def callback(self, left_img_msg, right_img_msg, lidar_msg, detected_objects_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')

        img_left_path = os.path.join(IMG_LEFT_DIR, f'{self.counter:06d}.png')
        img_right_path = os.path.join(IMG_RIGHT_DIR, f'{self.counter:06d}.png')
        cv2.imwrite(img_left_path, left_img)
        cv2.imwrite(img_right_path, right_img)

        lidar_bin_path = os.path.join(LIDAR_DIR, f'{self.counter:06d}.bin')
        self.save_lidar_to_bin(lidar_msg, lidar_bin_path)

        label_path = os.path.join(LABEL_DIR, f'{self.counter:06d}.txt')
        self.save_kitti_labels(detected_objects_msg, label_path)

        self.get_logger().info(f"Saved frame {self.counter}")
        self.counter += 1

    '''
    def callback(self, left_img_msg, right_img_msg, detected_objects_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')

        img_left_path = os.path.join(IMG_LEFT_DIR, f'{self.counter:06d}.png')
        img_right_path = os.path.join(IMG_RIGHT_DIR, f'{self.counter:06d}.png')
        cv2.imwrite(img_left_path, left_img)
        cv2.imwrite(img_right_path, right_img)

        label_path = os.path.join(LABEL_DIR, f'{self.counter:06d}.txt')
        self.save_kitti_labels(detected_objects_msg, label_path)

        self.get_logger().info(f"Saved frame {self.counter}")
        self.counter += 1'''

    def save_lidar_to_bin(self, msg, filename):
        points = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([p[0], p[1], p[2], p[3]])
        points_np = np.array(points, dtype=np.float32)
        points_np.tofile(filename)

    def get_yaw_from_quaternion(self, q):
        # Quaternion to Euler: roll (x), pitch (y), yaw (z)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def convert_base_link_to_camera(self, points, transform):
        """
        Convert LiDAR points to camera coordinates using a transformation matrix.
        :param points: Nx3 array of LiDAR points (x, y, z)
        :param transform: 4x4 transformation matrix from LiDAR to camera
        :return: Nx3 array of transformed points in camera coordinates
        """
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = transform @ points_homogeneous.T
        return transformed_points[:3, :].T

    def convert_base_link_to_camera_dim(self, dims, rotation_matrix):
        """
        Convert dimensions from LiDAR to camera coordinates.
        :param dims: Dimensions in LiDAR coordinates (length, width, height)
        :param rotation_matrix: 3x3 rotation matrix for the transformation
        :return: Dimensions in camera coordinates (length, width, height)
        """
        # Apply rotation to the dimensions
        new_dims = np.dot(rotation_matrix, np.array(dims))
        return new_dims
    def compute_box_3d(self, rotation_y, location, dimensions):
        l, w, h = dimensions
        x, y, z = location
        # 3D bounding box corners centered at origin (geometric center)
        x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
        y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        corners = np.array([x_corners, y_corners, z_corners])
        # Rotation matrix around Y axis
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])
        # Rotate corners
        corners_3d = np.dot(R, corners)
        # Translate corners to the object's location
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        return corners_3d
    def project_to_image(self, corners_3d, P2):
        """
        Project 3D corners to 2D image plane using camera matrix P2.
        :param corners_3d: 3D corners of the bounding box (shape: 3x8)
        :param P2: Camera projection matrix (shape: 3x4)
        :return: 2D corners in image coordinates (shape: 2x8)
        """
        # Convert corners to homogeneous coordinates
        ones = np.ones((1, corners_3d.shape[1]))
        corners_3d_homogeneous = np.vstack((corners_3d, ones))
        # Project to image plane
        corners_2d_homogeneous = P2 @ corners_3d_homogeneous
        # Normalize by the third coordinate
        corners_2d = corners_2d_homogeneous[:2, :] / corners_2d_homogeneous[2, :]
        return corners_2d
    def compute_2d_bbox(self, corners_2d):
        """
        Compute 2D bounding box from projected 3D corners.
        :param corners_2d: 2D corners of the bounding box (shape: 2x8)
        :return: Bounding box in format [left, top, right, bottom]
        """
        x_min = np.min(corners_2d[0, :])
        x_max = np.max(corners_2d[0, :])
        y_min = np.min(corners_2d[1, :])
        y_max = np.max(corners_2d[1, :])
        return [x_min, y_min, x_max, y_max]
    def compute_alpha(self, location, rotation_y):
        """
        Compute the alpha angle for KITTI format.
        :param location: 3D location of the object [x, y, z]
        :param rotation_y: Rotation around Y-axis in radians
        :return: Alpha angle in radians
        """
        x, _, z = location
        alpha = rotation_y - np.arctan2(x, z)
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        return float(alpha)

    def save_kitti_labels(self, msg, filename):
        lines = []
        for obj in msg.objects:
            #TODO: Take the transformation from camera0/camera_optical_link to base_link
            BL_to_Cam_T = np.array([[-0.036, -0.999, 0.001, 0.031],
                 [-0.015, -0.000, -1.000, 1.913],
                 [0.999, -0.036, -0.015, 0.0],
                 [0.000, 0.000, 0.000, 1.000]])
            #TODO: Take the P2 matrix from camera_info of left camera
            P2 = np.array([[960.0, 0.000000e+00, 960.5, 259.20001220703125],
                  [0.000000e+00, 959.3908081054688, 540.5, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
            #print(obj.classification[0].label)
            if obj.classification[0].label == 1:
                label = 'Car'
            elif obj.classification[0].label == 2 or obj.classification[0].label == 3 or obj.classification[0].label == 4:
                label = 'Truck'
            else:
                label = 'Other'

            # Placeholder values for 2D bounding box and truncation/truncation values
            truncated = -1
            occluded = -1


            #print(f"Object Existence Probability: {obj.existence_probability}")
            # Dimensions: height, width, length (KITTI expects h, w, l)
            dims = obj.shape.dimensions
            l, w, h = dims.x, dims.y, dims.z  # KITTI: h,w,l in camera coords
            # Convert dimensions from base_link to camera coordinates
            l_cam, w_cam, h_cam = self.convert_base_link_to_camera_dim([l, w, h], BL_to_Cam_T[:3, :3])
            l_cam = np.abs(l_cam)  # Ensure positive dimensions
            w_cam = np.abs(w_cam)
            h_cam = np.abs(h_cam)
            #print(f"Dimensions in camera coords: l={l_cam}, w={w_cam}, h={h_cam}")
            # Position in base_link coordinate frame
            pos = obj.kinematics.pose_with_covariance.pose.position
            x, y, z = pos.x, pos.y, pos.z
            # Convert position from base_link to camera coordinates
            x_cam, y_cam, z_cam = self.convert_base_link_to_camera(np.array([[x, y, z]]), BL_to_Cam_T)[0]

            # Yaw angle around Y axis (KITTI expects rotation_y)
            orientation = obj.kinematics.pose_with_covariance.pose.orientation
            rotation_y = -self.get_yaw_from_quaternion(orientation)#Minus because Y axis is flipped in KITTI compared to base_link
            corners_3d = self.compute_box_3d(rotation_y, [x_cam, y_cam, z_cam], [l_cam, w_cam, h_cam])
            #print(f"3D corners in camera coords: {corners_3d}")
            corners_2d = self.project_to_image(corners_3d, P2)
            bbox_2d = self.compute_2d_bbox(corners_2d)
            # Find 2d BBox coordinates
            bbox_left = bbox_2d[0]
            bbox_top = bbox_2d[1]
            bbox_right = bbox_2d[2]
            bbox_bottom = bbox_2d[3]

            alpha = self.compute_alpha([x_cam, y_cam, z_cam], rotation_y)

            line = f"{label} {truncated} {occluded} {alpha:.4f} {bbox_left:.4f} {bbox_top:.4f} {bbox_right:.4f} {bbox_bottom:.4f} " \
                   f"{h_cam:.6f} {w_cam:.6f} {l_cam:.6f} {x_cam:.6f} {y_cam:.6f} {z_cam:.6f} {rotation_y:.8f}"
            if bbox_left < 0 or obj.existence_probability < 0.5 or z_cam < 0:
                continue
            lines.append(line)

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

def main(args=None):
    rclpy.init(args=args)
    node = AwsimToKittiDataset()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
