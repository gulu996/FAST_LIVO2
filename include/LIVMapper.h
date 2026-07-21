/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "gnss_manager.h"
#include "vio.h"
#include "preprocess.h"
#include "uwb_manager.h"
#include "voxel_filter_utils.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <netinet/in.h>
#include <vikit/camera_loader.h>

#include <cstdint>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents(ros::NodeHandle &nh);
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void applyUwbUpdate(const char *stage);
  void applyGnssUpdate(const char *stage);
  bool publishRawBackendOdometry();
  void handleUwbRelocalizationConfirmed(UwbUpdateResult &result, const char *stage);
  void advanceUwbOutputCorrection();
  V3D outputPosition() const;
  void savePCD();
  void print_landmarks();
  void processImu();
  bool shouldSelectVisualFrame();
  void updateVisualObservationHints();
  void updateRuntimeGuard(double frame_time_s);
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes,const ros::Publisher &pubLaserCloudMap, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void initializeUdpReporter();
  void sendUdpMessage(const std::string &message);
  void sendUdpPose(const Eigen::Vector3d &position);
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double voxel_lidar_max_range_m_ = 450.0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  bool save_log_en = true;
  bool uwb_output_correction_en_ = false;
  bool uwb_output_smooth_en_ = true;
  double uwb_output_smooth_alpha_ = 0.15;
  double uwb_output_smooth_max_step_m_ = 0.05;
  int external_update_pause_map_frames_ = 0;
  int external_update_pause_map_frames_after_correction_ = 3;
  double external_update_pause_map_min_correction_m_ = 0.05;
  V3D uwb_output_pos_offset_ = V3D::Zero();
  V3D uwb_output_target_offset_ = V3D::Zero();
  bool pos_output_enable_timestamp_ = true;
  std::string pos_output_format_ = "timestamp_xyz_quat";
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;
  int pub_scan_num_nominal_ = 1;
  int pub_scan_num_degraded_ = 4;
  string save_path;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  bool dense_map_en_nominal_ = false;
  bool pcd_save_en_nominal_ = false;
  bool colorize_cloud_en_ = true;
  bool colorize_cloud_en_nominal_ = true;
  int publish_img_stride_ = 1;
  int publish_img_counter_ = 0;
  int lio_map_update_stride_ = 1;
  int lio_map_update_counter_ = 0;
  bool print_console_timing_en_ = true;
  int print_console_timing_stride_ = 1;
  bool suppress_image_pub_ = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  bool visual_map_prune_en = true;
  int visual_map_max_voxels = 1800;
  int visual_map_max_points_per_voxel = 10;
  int visual_map_max_total_points = 20000;
  int visual_map_max_add_per_frame_ = 300;
  double visual_map_min_shi_tomasi_score_ = 10.0;
  int pcd_cache_max_points = 300000;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  bool aruco_landmarks_en = false;
  bool udp_report_en = false;
  std::string udp_target_ip_;
  int udp_report_port_ = 0;
  std::string udp_device_id_;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  int vio_min_retrieve_points_ = 45;
  int vio_min_update_meas_ = 900;
  int vio_low_track_force_update_stride_ = 0;
  int vio_low_track_force_min_points_ = 8;
  bool vio_deterministic_visual_update_en_ = true;
  bool deterministic_state_snap_en_ = true;
  bool deterministic_pixel_snap_en_ = true;
  bool deterministic_camera_point_snap_en_ = true;
  bool deterministic_contiguous_image_copy_en_ = true;
  bool deterministic_imu_accept_out_of_order_en_ = true;
  bool deterministic_imu_buffer_sort_en_ = true;
  bool deterministic_prop_imu_buffer_sort_en_ = true;
  bool deterministic_image_buffer_sort_en_ = true;
  bool deterministic_sync_wait_for_image_lookahead_en_ = true;
  bool deterministic_pending_vio_image_en_ = true;
  bool deterministic_lio_feature_sort_en_ = true;
  bool deterministic_visual_observed_voxel_sort_en_ = true;
  bool deterministic_visual_voxel_key_sort_en_ = true;
  bool vio_visual_update_guard_en_ = true;
  double vio_visual_update_max_trans_m_ = 0.12;
  double vio_visual_update_max_rot_deg_ = 2.0;
  double vio_visual_update_max_backward_m_ = 0.03;
  double vio_visual_update_max_backward_ratio_ = 0.08;
  double vio_visual_update_backward_abs_floor_m_ = 0.003;
  double vio_visual_update_max_lateral_m_ = 0.08;
  double vio_visual_update_max_lateral_ratio_ = 0.35;
  double vio_visual_update_max_exposure_delta_ = 0.30;
  bool vio_image_quality_gate_en_ = false;
  double vio_image_quality_max_saturated_fraction_ = 0.20;
  double vio_image_quality_max_tile_saturated_fraction_ = 0.35;
  double vio_image_quality_max_dark_fraction_ = 0.98;
  double vio_image_quality_min_intensity_std_ = 6.0;
  bool vio_visual_patch_quality_gate_en_ = true;
  double vio_visual_patch_max_saturated_fraction_ = 0.10;
  double vio_visual_patch_min_intensity_std_ = 2.0;
  int vio_image_quality_saturated_pixel_value_ = 250;
  int vio_image_quality_dark_pixel_value_ = 5;
  int vio_image_quality_tile_rows_ = 4;
  int vio_image_quality_tile_cols_ = 4;
  double outlier_threshold;
  double vio_max_state_update_rot_deg_ = 0.8;
  double vio_max_state_update_trans_m_ = 0.08;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  cv::Mat pending_vio_img_;
  double pending_vio_time_ = 0.0;
  bool has_pending_vio_img_ = false;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV = 100.0;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;
  UwbManagerPtr uwb_manager;
  GnssManagerPtr gnss_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubRawBackendOdom;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  ros::Publisher mavros_pose_publisher;
  std::string raw_backend_odom_topic_ = "/backend/livo_odom_raw";
  std::string raw_backend_odom_frame_id_ = "odom";
  std::string raw_backend_body_frame_id_ = "body";
  std::int64_t last_raw_backend_odom_stamp_ns_ = -1;
  std::uint64_t raw_backend_odom_attempted_ = 0;
  std::uint64_t raw_backend_odom_published_ = 0;
  std::uint64_t raw_backend_odom_duplicate_ = 0;
  std::uint64_t raw_backend_odom_non_monotonic_ = 0;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
  bool global_map_pub = false;  
  int udp_socket_fd_ = -1;
  struct sockaddr_in udp_target_addr_ {};
  bool udp_socket_ready_ = false;

  bool adaptive_visual_selector_en = true;
  double keyframe_trans_thresh_min_ = 0.08;
  double keyframe_trans_thresh_max_ = 0.18;
  double keyframe_rot_thresh_min_deg_ = 1.2;
  double keyframe_rot_thresh_max_deg_ = 2.5;
  double keyframe_constraint_ratio_full_ = 0.2;
  int keyframe_max_skip_frames_ = 4;
  double keyframe_trans_thresh_min_nominal_ = 0.08;
  double keyframe_trans_thresh_max_nominal_ = 0.18;
  double keyframe_rot_thresh_min_deg_nominal_ = 1.2;
  double keyframe_rot_thresh_max_deg_nominal_ = 2.5;
  int keyframe_max_skip_frames_nominal_ = 4;

  bool runtime_guard_en_ = true;
  double frame_time_budget_s_ = 0.1;
  int runtime_over_budget_trigger_frames_ = 2;
  int runtime_recover_trigger_frames_ = 8;
  int vio_max_iterations_nominal_ = 5;
  int vio_max_iterations_degraded_ = 2;
  double keyframe_trans_scale_degraded_ = 1.8;
  double keyframe_rot_scale_degraded_ = 1.8;
  int keyframe_max_skip_frames_degraded_ = 8;
  bool disable_dense_map_in_degraded_ = true;
  bool disable_pcd_save_in_degraded_ = true;
  bool disable_colorize_cloud_in_degraded_ = false;
  bool disable_image_publish_in_degraded_ = true;
  int runtime_over_budget_count_ = 0;
  int runtime_under_budget_count_ = 0;
  bool runtime_degraded_mode_ = false;

  int skipped_visual_frames_ = 0;
  bool has_last_visual_keyframe_state_ = false;
  StatesGroup last_visual_keyframe_state_;
  std::string last_selector_reason_ = "init";
  double last_selector_constraint_ratio_ = 0.0;
  double last_selector_trans_thresh_ = 0.0;
  double last_selector_rot_thresh_deg_ = 0.0;
  double last_selector_trans_delta_ = 0.0;
  double last_selector_rot_delta_deg_ = 0.0;
  bool last_selector_reach_pose_keyframe_ = false;
  bool last_selector_reach_skip_limit_ = false;

  int sub_lidar_queue_size_ = 128;
  int sub_imu_queue_size_ = 512;
  int sub_img_queue_size_ = 16;
  int max_lidar_buffer_size_ = 32;
  int max_imu_buffer_size_ = 3000;
  int max_img_buffer_size_ = 12;
  int max_prop_imu_buffer_size_ = 3000;
  int sync_img_buffer_min_size_ = 1;
  double sync_img_lookahead_time_ = 0.0;
};
#endif
