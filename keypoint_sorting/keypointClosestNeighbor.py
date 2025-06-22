#!/usr/bin/env python3
# This function processes the incoming messages, computes 3D poses, and publishes
import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseArray
import sensor_msgs_py.point_cloud2 as pc2 
import std_msgs.msg

# Assuming srd_interfaces is a custom ROS2 package you have
from srd_interfaces.msg import InferenceStamped


class KeypointClosestNeighbor(Node):
    """
    A ROS2 node that processes synchronized keypoint detections from left and right cameras,
    computes 3D poses (or related spatial information), and publishes a PoseArray
    and a PointCloud2. It also tracks previous keypoints for potential
    temporal consistency or closest neighbor matching (though the matching logic
    is not present in this snippet).
    """

    # Define a class-level constant for the queue size for ROS2 publishers/subscribers.
    # This helps manage the buffer for incoming/outgoing messages.
    QUEUE_SIZE = 20

    def __init__(self):
        """
        Initializes the KeypointClosestNeighbor node, setting up ROS2 publishers,
        subscribers, and message synchronization.
        """
        # Call the constructor of the parent class (Node) and name this node.
        super().__init__('Sorted_PointCloud2_publisher')
        self.get_logger().info('KeypointClosestNeighbor node starting...')

        # --- Member Variables ---
        # Store previous keypoints. This is use to calculate the 3d poses from 
        # the previous keypoints and the current keypoints.
        self.prev_keypoints = None

        # --- ROS2 Publishers ---
        # Publisher for an array of poses. This will likely contain the 3D poses
        # calculated from the synchronized stereo keypoints.
        self.pose_publisher = self.create_publisher(
            PoseArray,          # Message type
            '/multi_pose_cones', # ROS2 topic name
            self.QUEUE_SIZE     # Queue size
        )
        self.get_logger().info(f'Publishing PoseArray to {self.pose_publisher.topic_name}')


        # Publisher for a PointCloud2 message. This represents the 3D locations
        # of the detected cones.
        self.publisher_pointcloud = self.create_publisher(
            PointCloud2,         # Message type
            '/pointcloud2_cones', # ROS2 topic name
            self.QUEUE_SIZE      # Queue size
        )
        self.get_logger().info(f'Publishing Cone Poses as PointCloud2 to {self.publisher_pointcloud.topic_name}')


        # Publisher for a PointCloud2 message. This represents the 3D locations
        # of the detected cones during motion.
        self.publisher_pointcloud_motion = self.create_publisher(
            PointCloud2,         # Message type
            '/pointcloud2_cones_with_motion', # ROS2 topic name
            self.QUEUE_SIZE      # Queue size
        )
        self.get_logger().info(f'Publishing Cone Poses as PointCloud2 to {self.publisher_pointcloud.topic_name}')


        # --- ROS2 Subscribers (using message_filters for synchronization) ---

        # Subscriber for inference results (keypoints, bounding boxes, etc.) from the left camera.
        self._detections_subscription_left = message_filters.Subscriber(
            self,                      # Node instance
            InferenceStamped,          # Message type
            '/left_camera/detections'  # ROS2 topic name
        )
        self.get_logger().info(f'Subscribing to left detections: {self._detections_subscription_left.topic}')

        # Subscriber for inference results from the right camera.
        self._detections_subscription_right = message_filters.Subscriber(
            self,                       # Node instance
            InferenceStamped,           # Message type
            '/right_camera/detections'  # ROS2 topic name
        )
        self.get_logger().info(f'Subscribing to right detections: {self._detections_subscription_right.topic}')

        # Subscriber for the right camera's calibration information. This is crucial
        # for stereo triangulation and projecting 2D keypoints into 3D.
        # Note: It's good practice to also subscribe to the right camera info if needed
        # for full stereo rectification/triangulation, or ensure that only one is
        # necessary for your specific 3D reconstruction approach.
        self.left_camera_parameters = message_filters.Subscriber(
            self,                               # Node instance
            CameraInfo,                         # Message type
            '/left_camera/resize/camera_info'  # ROS2 topic name
        )
        self.get_logger().info(f'Subscribing to right camera info: {self.left_camera_parameters.topic}')

        # --- Message Synchronization ---
        # TimeSynchronizer groups messages from multiple topics that arrive within
        # a short time window. This is essential for stereo vision applications
        # where corresponding left and right camera data (and camera info)
        # must be processed together.
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [
                self._detections_subscription_left,  # Left camera detections
                self._detections_subscription_right, # Right camera detections
                self.left_camera_parameters         # Left camera intrinsic parameters
            ],
            self.QUEUE_SIZE  # The queue size for synchronizer
        )
        self.get_logger().info('TimeSynchronizer created for left detections, right detections, and left camera info.')


        # Register the callback function that will be triggered when a synchronized
        # set of messages arrives.
        self.time_synchronizer.registerCallback(self.detections_callback)
        self.get_logger().info('Registered detections_callback with TimeSynchronizer.')

    def detections_callback(self, detections_msg_left: InferenceStamped,
                            detections_msg_right: InferenceStamped,
                            camera_info_msg: CameraInfo):
        """
        Callback function for synchronized detection and camera info messages.
        It processes stereo detections to estimate 3D poses and generate a point cloud.

        Args:
            detections_msg_left (InferenceStamped): Detections from the left camera.
            detections_msg_right (InferenceStamped): Detections from the right camera.
            camera_info_msg (CameraInfo): Camera calibration parameters (specifically K matrix).
                                          Note: This is assumed to be the left camera's
                                          parameters if used directly for disparity,
                                          or adjusted if it's the right camera's.
        """
        self.get_logger().debug('Received synchronized messages.')

        # Initialize lists to store centroid positions for matching
        left_centroids = []
        right_centroids = []

        # Populate centroid lists from the received detection messages
        # Assuming `detections_msg.objects` contains a list of detections
        # where each detection has a `bbox` with a `centroid.x` and `centroid.y`.
        for detection_left, detection_right in zip(detections_msg_left.objects, detections_msg_right.objects):
            left_centroids.append((detection_left.bbox.centroid.x, detection_left.bbox.centroid.y))
            right_centroids.append((detection_right.bbox.centroid.x, detection_right.bbox.centroid.y))

        # --- Input Validation ---
        if not left_centroids or not right_centroids or len(left_centroids) != len(right_centroids):
            self.get_logger().warn(
                "Left and right detections do not match in count or are empty. Skipping 3D reconstruction."
            )
            self.prev_keypoints = detections_msg_left # Still update for next iteration if needed
            return

        # Convert lists to numpy arrays for efficient computation and manipulation
        left_centroids_np = np.array(left_centroids, dtype=np.float32)
        right_centroids_np = np.array(right_centroids, dtype=np.float32)

        # Create a new array to store re-ordered right positions after matching.
        # This array will hold the right centroid that best matches each left centroid.
        matched_right_centroids_np = np.zeros_like(left_centroids_np)
        
        # Keep a copy of right centroids that are available for matching.
        # As matches are found, they are removed from this pool.
        unmatched_right_centroids_pool = right_centroids_np.copy()

        # --- Closest Neighbor Matching ---
        # This loop iterates through each left centroid and finds its closest neighbor
        # among the currently available right centroids (those not yet matched).
        # This is a greedy closest neighbor approach.
        index_pool = []
        for i, left_pos in enumerate(left_centroids_np):
            # Calculate Euclidean distances from the current left centroid to all
            # unmatched right centroids.
            distances = np.linalg.norm(unmatched_right_centroids_pool - left_pos, axis=1)
            
            # Find the index of the closest right centroid in the `unmatched_right_centroids_pool`.
            # closest_index_in_pool = np.argmin(distances)
            closest_index_in_pool = np.argsort(distances)[0]
            # Ensure the closest index is not already in the index pool to avoid re-matching.
            # This is a safeguard against potential duplicates or re-matching.
            # If the closest index is already in the pool, find the next closest one.
            # This loop ensures that we always find a unique index for matching.
            least_index = 1
            while closest_index_in_pool in index_pool:
                closest_index_in_pool = np.argsort(distances)[least_index]
                least_index += 1 
            index_pool.append(closest_index_in_pool)
            # Assign the matched right centroid to its corresponding position in the
            # `matched_right_centroids_np` array.
            matched_right_centroids_np[i] = unmatched_right_centroids_pool[closest_index_in_pool]
            
            # Remove the matched right centroid from the pool to prevent re-matching.
            # unmatched_right_centroids_pool = np.delete(
            #     unmatched_right_centroids_pool, closest_index_in_pool, axis=0
            # )

        # Update the objects in the `detections_msg_right` to reflect the
        # matched order. This step modifies the incoming message, which might
        # be unexpected if `detections_msg_right` is used elsewhere.
        # A safer approach might be to just use `matched_right_centroids_np`
        # directly in subsequent calculations without modifying the message.
        # print(f"Matched indices: {index_pool}")
        old_detections_right = [detection for detection in detections_msg_right.objects]
        for i in range(len(matched_right_centroids_np)):
            detections_msg_right.objects[i] =  old_detections_right[index_pool[i]]

        # --- Disparity Calculation ---
        # Calculate disparity based on the *mean* x-coordinate of keypoints for each object.
        # This assumes that each object detection has a list of keypoints.
        disparities = []
        for detection_left, detection_right in zip(detections_msg_left.objects, detections_msg_right.objects):
            # Ensure there are keypoints to process
            if not detection_left.keypoints or not detection_right.keypoints:
                self.get_logger().warn("Missing keypoints in one or more detections. Skipping disparity for this pair.")
                continue

            # Calculate the mean x-coordinate for keypoints in each detection
            x_values_left_mean = np.mean([point.x for point in detection_left.keypoints])
            x_values_right_mean = np.mean([point.x for point in detection_right.keypoints])
            
            # Disparity is typically `x_left - x_right`.
            # A larger positive disparity means the object is closer.
            disparity_val = x_values_left_mean - x_values_right_mean
            disparities.append(disparity_val)

        if not disparities:
            self.get_logger().warn("No valid disparities calculated. Skipping 3D reconstruction.")
            self.prev_keypoints = detections_msg_left
            return

        disparities_np = np.array(disparities)

        # --- Depth Calculation ---
        # Fixed baseline for stereo camera setup (e.g., distance between camera optical centers).
        # This value (0.12 meters) must be accurate to the stereo camera uesd.
        BASELINE = 0.12 # meters
        
        # Focal length in pixels from the camera intrinsic matrix (K).
        # camera_info_msg.k is a (9,) matrix: [fx 0 cx 0 fy cy 0 0 1]
        # K[0] corresponds to fx (focal length in x-direction).
        focal_length_px = camera_info_msg.k[0] 
        
        # Compute depth for all detected objects.
        depth_np = self.compute_depth(disparities_np, BASELINE, focal_length_px)

        # --- 3D Reconstruction and Publishing ---
        # Only proceed if previous keypoints are available for temporal association.
        # The `reconstrucr_3d` method (note the typo 'reconstrucr' -> 'reconstruct')
        # is expected to use the calculated depths and the current/previous keypoints
        # to generate 3D poses.
        if self.prev_keypoints is not None:
            # Reconstruct 3D poses using current depths, left detections,
            # previous keypoints, and camera parameters.
            # Assuming `reconstrucr_3d` returns a PoseArray and a NumPy array of 3D points.
            pose_array_msg, np_3d_points = self.reconstruct_3d(
                depth_np, detections_msg_left, camera_info_msg
            )

            # --- Publish PoseArray ---
            self.pose_publisher.publish(pose_array_msg)
            # self.get_logger().info(f'Published PoseArray with {len(pose_array_msg.poses)} poses.')

            # --- Publish PointCloud2 ---
            # Create the header for the PointCloud2 message
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            # The frame_id should be consistent with your TF tree.
            # 'camera_link' or 'base_link' of the camera system are common choices.
            # 'map' is typically for global coordinates, which might require a transform.
            # header.frame_id = camera_info_msg.header.frame_id or 'camera_link'
            header.frame_id = 'map'  # Adjust as necessary for TF setup

            # Create the PointCloud2 message from the NumPy array of 3D points.
            # `pc2.create_cloud_xyz32` expects a list of lists or a list of tuples for points.
            np_3d_points = np_3d_points.reshape(-1, 3)  # Ensure shape is (N, 3)
            np_3d_points[:,1] = np_3d_points[:,2]  # Set Y to Z for PointCloud2
            np_3d_points[:,2] = 0  # Set Z to 0 for PointCloud2 (if needed, adjust based on your coordinate system)
            cloud_msg = pc2.create_cloud_xyz32(header, np_3d_points.tolist())
            self.publisher_pointcloud.publish(cloud_msg)
            self.get_logger().info(f'Published PointCloud2 with {len(np_3d_points)} points.')

        else:
            self.get_logger().info("No previous keypoints found, skipping 3D reconstruction and publication.")
        
        # Store the current left detections as previous keypoints for the next iteration.
        self.prev_keypoints = detections_msg_left

    def compute_depth(self, disparity: np.ndarray, baseline: float, fx: float) -> np.ndarray:
        """
        Computes the depth for each point given its disparity.

        The formula used is: Depth = (Baseline * Focal_Length_X_Pixels) / Disparity

        Args:
            disparity (np.ndarray): A NumPy array of disparity values (in pixels).
                                    (x_left - x_right) for corresponding points.
            baseline (float): The physical baseline distance between the two cameras in meters.
            fx (float): The focal length of the camera in the x-direction (in pixels).
                        This is K[0] from the camera intrinsic matrix.

        Returns:
            np.ndarray: A NumPy array of computed depths in meters, corresponding to the input disparities.
                        Invalid depths (e.g., from zero or negative disparity) are set to 0.
        """
        # Suppress "divide by zero" warnings temporarily as we handle them explicitly.
        with np.errstate(divide='ignore'):
            # Add a small epsilon to disparity to prevent true division by zero for very small values.
            # However, for zero/negative disparities, we explicitly set depth to 0 later.
            depth = (fx * baseline) / (disparity + 1e-6) # Added 1e-6 for numerical stability

            # Filter out invalid depths:
            # If disparity is zero or negative, it means the point is at infinity or invalid.
            # Setting depth to 0 signifies an invalid or uncomputable depth in this context.
            depth[disparity <= 0] = 0
            # depth = depth[depth > 0]  # Remove any negative or zero depths from the array
            
            # Optionally, you might also want to filter out extremely large depths if they are
            # physically impossible in your scene
            # depth[depth > MAX_REASONABLE_DEPTH] = 0
            
        print(f"Computed depths: min={np.min(depth) if depth.size > 0 else 'N/A'}, max={np.max(depth) if depth.size > 0 else 'N/A'}")
        return depth
    
    def reconstruct_3d(self, depth: np.ndarray, detections_msg_cur: InferenceStamped,
                       camera_info: CameraInfo, MAX_REASONABLE_DEPTH: float = 50)-> tuple[PoseArray, np.ndarray]:
        """
        Reconstructs 3D points from 2D image coordinates and calculated depths.
        It also filters out points based on a maximum depth threshold and
        generates a PoseArray and a NumPy array of 3D points.

        Args:
            depth (np.ndarray): A NumPy array of depth values (Z-coordinate) for each detected object.
            detections_msg_cur (InferenceStamped): Current frame's left camera detections.
                                                  Used for current 2D image coordinates (u, v).
            detections_msg_prev (InferenceStamped): Previous frame's left camera detections.
                                                   (Appears to be unused in the provided logic for 3D reconstruction,
                                                   might be a remnant from a tracking/association idea or a misunderstanding
                                                   of its intended role here. The current code uses `detection_cur` for `x_img`, `y_img`).
            camera_info (CameraInfo): Camera calibration parameters (K matrix) for unprojection.
                                      Assumed to be from the camera corresponding to `detections_msg_cur`.
            max_depth (float): Maximum allowable depth in meters. Points beyond this depth are
                                considered outliers and excluded from the output.

        Returns:
            tuple[PoseArray, np.ndarray]: A tuple containing:
                - PoseArray: A ROS2 PoseArray message populated with the 3D positions of the detected objects.
                - np.ndarray: A NumPy array of shape (N, 3) containing the X, Y, Z coordinates
                              of the reconstructed 3D points.
        """
        self.get_logger().debug(f"Starting 3D reconstruction for {len(depth)} detections.")

        reconstructed_points_3d = [] # List to accumulate 3D points as [x, y, z]
        poses = [] # List to accumulate Pose messages
        outliers = []

        # Extract camera intrinsics from the K matrix for unprojection.
        # K = [fx  0 cx
        #      0  fy cy
        #      0  0  1]
        fx = camera_info.k[0]
        fy = camera_info.k[4] 
        cx = camera_info.k[2]
        cy = camera_info.k[5] 

        # Ensure that `detections_msg_cur.objects` and `depth` have matching lengths.
        # The loop iterates based on `depth`, which implies a 1:1 correspondence.
        # Also, the original code zipped `detections_msg_prev.objects` but only `detections_msg_cur`
        # was used for `x_img`, `y_img`. I'll adjust the zip to only use relevant messages.
        for idx, (detection_cur, current_depth) in enumerate(zip(detections_msg_cur.objects, depth)):
            # --- Depth Filtering ---
            # Filter out points with depths outside the valid range [1, max_depth] meters.
            # if not (1 <= current_depth <= max_depth):
            #     self.get_logger().debug(f"Skipping point {idx} due to invalid depth: {current_depth:.2f}m (must be within [1, {max_depth}]).")
            #     continue # Skip to the next detection

            # Ensure keypoints exist before trying to process them
            if not detection_cur.keypoints:
                self.get_logger().warn(f"Detection {idx} has no keypoints. Skipping 3D reconstruction for this detection.")
                continue
            
            # We will not consider depth greater than max_depth
            if current_depth > MAX_REASONABLE_DEPTH or current_depth <= 0.2:  # Assuming a minimum depth of 0.2m to avoid noise
                outliers.append(idx)
                continue
            # print("Current index:", idx, "with depth:", current_depth)
            # Get the mean (u, v) image coordinates from the current detection's keypoints.
            # These are the 2D pixel coordinates in the image plane of the left camera.
            u_img = np.mean(np.array([point.x for point in detection_cur.keypoints]), axis=0)
            v_img = np.mean(np.array([point.y for point in detection_cur.keypoints]), axis=0)

            # --- 3D Unprojection ---
            # Convert 2D pixel coordinates (u, v) and depth (Z) into 3D camera coordinates (X, Y, Z).
            # The formulas for unprojection are:
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            x_3d = (u_img - cx) * current_depth / fx
            y_3d = (v_img - cy) * current_depth / fy
            z_3d = current_depth # Z-coordinate is the depth itself

            self.get_logger().debug(f"Reconstructed 3D point {idx}: (X={x_3d:.2f}, Y={y_3d:.2f}, Z={z_3d:.2f})")
            # print(f"Reconstructed 3D point {idx}: (X={x_3d:.2f}, Y={y_3d:.2f}, Z={z_3d:.2f})")
            
            reconstructed_points_3d.append([x_3d, y_3d, z_3d])

            # Create a Pose message for this 3D point
            pose = Pose()
            pose.position.x = x_3d
            pose.position.y = y_3d
            pose.position.z = z_3d
            # Assuming no orientation is estimated, set a default identity quaternion.
            pose.orientation.w = 1.0
            poses.append(pose)

        # Convert the list of 3D points to a NumPy array
        np_3d_points = np.array(reconstructed_points_3d, dtype=np.float32)
        # print(f"Total valid 3D points reconstructed: {len(np_3d_points)} with {len(outliers)} outliers removed.")
        # np_3d_points = np.delete(np_3d_points, np.array(outliers), axis=0)  # Remove outliers based on indices
        # Create the PoseArray message
        pose_array_msg = PoseArray()
        # The header should be consistent with the PointCloud2 header frame_id.
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = camera_info.header.frame_id or 'camera_link' # Consistent frame_id
        pose_array_msg.poses = poses # Assign the list of created poses

        self.get_logger().info(f"Finished 3D reconstruction. Generated {len(np_3d_points)} valid 3D cones coordinates from {len(detections_msg_cur.objects)} detected cones.")
        
        # The commented-out PnP part suggests an intention to calculate camera pose,
        # not necessarily individual object poses. If PnP is to be used,
        # `points_3D` would be 3D world points and `points_cur_2D` would be
        # their corresponding 2D image points, typically from a single camera view.
        # This part seems disconnected from the current object-wise 3D point generation.

        return pose_array_msg, np_3d_points


def main():
    rclpy.init()
    rclpy.spin(KeypointClosestNeighbor())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
