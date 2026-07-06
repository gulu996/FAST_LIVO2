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
#include "vio.h"
#include "preprocess.h"
#include "uwb_manager.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <netinet/in.h>
#include <vikit/camera_loader.h>

class LIVMapper
{
public:
  enum class SystemMode
  {
    NORMAL,
    DEGRADED_HOLD,
    DEGRADED_BOOTSTRAP,
    LOCAL_REINIT
  };

  enum class RejectReason
  {
    NONE,
    SAFETY,
    DEGRADED_HOLD,
    BACKWARD_SLIP,
    LOCAL_REINIT,
    BOOTSTRAP,
    SMALL_MOTION,
    NO_POINTS,
    UNKNOWN
  };

  struct ObservationQuality
  {
    bool lio_valid = true;
    bool vio_valid = true;
    bool uwb_valid = true;
    bool lio_degenerated = false;
    bool vio_low_tracked = false;
    bool map_update_weak = false;
    bool backward_slip = false;
    bool speed_abnormal = false;
    bool attitude_abnormal = false;
    bool uwb_residual_abnormal = false;
    double backward_distance = 0.0;
    double speed = 0.0;
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    std::string reason = "none";
  };

  struct UpdateDecision
  {
    bool allow_lio_update = true;
    bool allow_vio_update = true;
    bool allow_uwb_update = true;
    bool allow_voxel_map_update = true;
    bool allow_visual_map_update = true;
    bool allow_publish = true;
    SystemMode mode = SystemMode::NORMAL;
    RejectReason reason = RejectReason::NONE;
    std::string reason_text = "none";
  };

  struct MapUpdateDecision
  {
    bool allow = true;
    RejectReason reason = RejectReason::NONE;
    std::string skip_reason = "none";
  };

  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents(ros::NodeHandle &nh);
  void initializeFiles();
  void run();
  void gravityAlignment();
  void applyDegeneracyGuardCorrections(const char *stage = nullptr);
  void evaluateDegeneracyGuardUpdate(const char *stage,
                                     const StatesGroup &state_before_update,
                                     int lio_feature_count = -1,
                                     int visual_tracked_points = -1);
  bool applyScalarPseudoMeasurement(StatesGroup &state,
                                    const Eigen::Matrix<double, 1, DIM_STATE> &H,
                                    double residual,
                                    double sigma,
                                    double gain,
                                    double *correction_norm = nullptr);
  void blendDegeneracyGuardUpdate(const StatesGroup &state_before_update,
                                  const StatesGroup &state_after_update,
                                  double scale);
  double updateAdaptiveSensorNoiseScale(const char *sensor);
  void resetCorridorMotionPrior();
  void updateCorridorMotionPrior(const char *stage, const StatesGroup &state);
  void writeDegeneracyGuardLog(const char *stage = nullptr);
  bool isStateFiniteForSafety(const StatesGroup &state, const char *stage, std::string *reason = nullptr) const;
  bool validateStateForSafety(const char *stage,
                              const StatesGroup &state_before,
                              const StatesGroup &state_after,
                              bool update_backward_window,
                              bool allow_recover);
  void enterFailSafe(const std::string &reason, const StatesGroup *fallback_state = nullptr);
  void maybeRecoverFailSafe();
  void recordReliableStateForSafety(const char *stage);
  void clearSafetyLocalCaches();
  void clearLocalMapsForReinit(const std::string &reason);
  void enterLocalReinitMode(const std::string &mode, const std::string &reason);
  void updateLocalModeAtFrameStart();
  void updateLocalTrackingLostDetectors(const char *sensor);
  ObservationQuality evaluateObservationQuality() const;
  void updateSystemModeFromQuality(const ObservationQuality &quality);
  UpdateDecision makeUpdateDecision(const ObservationQuality &quality) const;
  MapUpdateDecision decideMapUpdate(bool lio_degenerated,
                                    bool stride_ready,
                                    bool force_ready,
                                    bool has_points) const;
  void printDiagnostics(const char *stage);
  static const char *systemModeName(SystemMode mode);
  static const char *rejectReasonName(RejectReason reason);
  bool isDegradedHoldMode() const;
  bool isBootstrapMode() const;
  void applyDegradedHoldConstraint(const char *stage, bool check_reject);
  bool localModeBlocksMapUpdate() const;
  bool localModeBlocksVisualMapUpdate() const;
  bool localModeSkipsVisualEkf() const;
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void applyUwbUpdate(const char *stage);
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
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  bool save_log_en = true;
  bool uwb_output_correction_en_ = true;
  bool uwb_output_smooth_en_ = true;
  double uwb_output_smooth_alpha_ = 0.15;
  double uwb_output_smooth_max_step_m_ = 0.05;
  V3D uwb_output_pos_offset_ = V3D::Zero();
  V3D uwb_output_target_offset_ = V3D::Zero();
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
  bool legacy_lock_z_after_gravity_align_en_ = false;

  bool deg_guard_enable_ = false;
  bool deg_guard_enable_z_soft_constraint_ = false;
  double deg_guard_z_ref_ = 0.0;
  double deg_guard_sigma_z_ = 0.20;
  double deg_guard_sigma_vz_ = 0.30;
  double deg_guard_z_gain_ = 1.0;
  bool deg_guard_enable_nhc_ = false;
  double deg_guard_sigma_body_vy_ = 0.05;
  double deg_guard_sigma_body_vz_ = 0.05;
  double deg_guard_nhc_min_speed_ = 0.05;
  bool deg_guard_nhc_only_in_degenerate_ = true;
  double deg_guard_nhc_gain_ = 1.0;
  bool deg_guard_enable_backward_guard_ = false;
  double deg_guard_backward_step_threshold_ = 0.05;
  double deg_guard_backward_speed_threshold_ = 0.20;
  int deg_guard_backward_consecutive_frames_ = 10;
  std::string deg_guard_backward_action_ = "log_only";
  bool deg_guard_enable_corridor_detection_ = false;
  int deg_guard_min_lidar_features_ = 30;
  int deg_guard_min_visual_tracked_points_ = 2;
  int deg_guard_vio_low_feature_tracked_points_ = 5;
  bool deg_guard_use_vio_skip_for_degenerate_ = false;
  bool deg_guard_use_vio_large_rotation_for_reject_ = false;
  double deg_guard_camera_dt_ = 1.0 / 30.0;
  double deg_guard_lidar_dt_ = 0.1;
  double deg_guard_max_update_translation_norm_ = 0.5;
  double deg_guard_max_update_yaw_deg_ = 5.0;
  double deg_guard_max_update_translation_rate_mps_ = 3.0;
  double deg_guard_max_update_yaw_rate_degps_ = 180.0;
  double deg_guard_hessian_condition_threshold_ = 1000.0;
  int deg_guard_min_degenerate_frames_ = 3;
  int deg_guard_recover_frames_ = 3;
  double deg_guard_degenerate_lio_noise_scale_ = 1.2;
  double deg_guard_degenerate_vio_noise_scale_ = 2.0;
  bool deg_guard_enable_adaptive_sensor_weighting_ = false;
  double deg_guard_adaptive_lio_base_noise_scale_ = 1.0;
  double deg_guard_adaptive_vio_base_noise_scale_ = 1.0;
  double deg_guard_adaptive_lio_low_feature_noise_scale_ = 2.0;
  double deg_guard_adaptive_vio_low_track_noise_scale_ = 20.0;
  double deg_guard_adaptive_lio_high_residual_noise_scale_ = 2.0;
  double deg_guard_adaptive_lio_residual_ref_ = 0.05;
  double deg_guard_adaptive_max_noise_scale_ = 10.0;
  int deg_guard_vio_skip_min_tracked_points_ = 2;
  int deg_guard_vio_skip_tracked_points_ = 1;
  int deg_guard_vio_min_update_meas_ = 32;
  bool deg_guard_reject_large_update_in_degenerate_ = false;
  bool deg_guard_reject_nonfinite_update_ = true;
  double deg_guard_max_degenerate_update_translation_ = 0.3;
  double deg_guard_max_degenerate_update_yaw_deg_ = 3.0;
  std::string deg_guard_log_file_ = "";
  bool deg_guard_corridor_degenerate_ = false;
  int deg_guard_degenerate_count_ = 0;
  int deg_guard_recover_count_ = 0;
  int deg_guard_backward_count_ = 0;
  bool deg_guard_last_pos_ready_ = false;
  V3D deg_guard_last_pos_ = V3D::Zero();
  V3D deg_guard_last_velocity_body_ = V3D::Zero();
  double deg_guard_last_z_residual_ = 0.0;
  double deg_guard_last_z_correction_norm_ = 0.0;
  double deg_guard_last_nhc_correction_norm_ = 0.0;
  double deg_guard_last_forward_progress_ = 0.0;
  double deg_guard_last_forward_progress_rate_mps_ = 0.0;
  bool deg_guard_last_backward_slip_ = false;
  int deg_guard_last_lio_feature_count_ = -1;
  int deg_guard_last_visual_tracked_points_ = -1;
  double deg_guard_last_update_translation_norm_ = 0.0;
  double deg_guard_last_update_yaw_deg_ = 0.0;
  double deg_guard_last_dt_ = 0.0;
  double deg_guard_last_lio_time_ = -1.0;
  double deg_guard_last_vio_time_ = -1.0;
  double deg_guard_last_update_translation_rate_mps_ = 0.0;
  double deg_guard_last_update_yaw_rate_degps_ = 0.0;
  double deg_guard_last_visual_update_rot_deg_ = 0.0;
  double deg_guard_last_visual_update_rot_rate_degps_ = 0.0;
  double deg_guard_last_final_pose_delta_ = 0.0;
  double deg_guard_last_lio_noise_scale_ = 1.0;
  double deg_guard_last_vio_noise_scale_ = 1.0;
  bool deg_guard_last_lio_downweighted_ = false;
  bool deg_guard_last_lio_update_executed_ = false;
  bool deg_guard_last_lio_voxel_map_updated_ = false;
  bool deg_guard_last_vio_skip_affects_degenerate_ = false;
  std::string deg_guard_last_lio_voxel_map_skip_reason_ = "not_lio";
  std::string deg_guard_last_sensor_type_ = "UNKNOWN";
  std::string deg_guard_last_weight_reason_ = "off";
  std::string deg_guard_last_update_status_ = "accepted";
  std::string deg_guard_last_reject_reason_ = "";
  std::string deg_guard_last_action_ = "none";
  std::string deg_guard_last_reason_ = "init";

  bool safety_guard_enable_ = false;
  bool safety_fail_safe_mode_ = false;
  int safety_fail_safe_stable_frames_ = 0;
  int safety_fail_safe_recover_frames_ = 10;
  double safety_max_speed_ = 3.0;
  double safety_max_frame_translation_ = 0.5;
  double safety_max_frame_rotation_deg_ = 15.0;
  double safety_backward_time_window_ = 5.0;
  double safety_backward_distance_threshold_ = 1.0;
  std::string safety_backward_action_ = "fail_safe_or_downweight";
  std::string safety_last_reason_ = "normal";
  double safety_last_speed_ = 0.0;
  double safety_last_quat_norm_ = 1.0;
  double safety_last_frame_translation_ = 0.0;
  double safety_last_frame_rotation_deg_ = 0.0;
  double safety_backward_distance_in_window_ = 0.0;
  std::deque<std::pair<double, double>> safety_backward_window_;
  bool safety_reliable_state_ready_ = false;
  StatesGroup safety_reliable_state_;
  bool skip_mapping_this_frame_ = false;
  bool deterministic_mode_ = true;
  int deterministic_frame_id_ = 0;
  bool deterministic_last_lio_update_ = false;
  bool deterministic_last_vio_update_ = false;
  bool deterministic_last_uwb_update_ = false;
  std::string deterministic_last_uwb_anchor_ids_ = "";
  double uwb_update_window_sec_ = 0.05;
  double uwb_relocalize_xy_threshold_ = 1.0;
  bool uwb_relocalize_en_ = false;
  bool uwb_update_only_on_lio_ = true;

  bool local_reinit_enable_ = true;
  bool debug_fixed_degraded_intervals_enable_ = false;
  bool degraded_bootstrap_enable_ = true;
  bool disable_visual_map_in_degraded_hold_ = true;
  bool disable_voxel_map_in_degraded_hold_ = true;
  std::string fixed_degraded_trigger_mode_ = "manual_time";
  double degraded_hold_attitude_reject_deg_ = 5.0;
  double degraded_hold_speed_reject_mps_ = 1.0;
  double bag_start_offset_ = 0.0;
  double fixed_degraded_first_start_sec_ = 150.0;
  double fixed_degraded_first_end_sec_ = 190.0;
  double fixed_degraded_second_start_sec_ = 520.0;
  double fixed_degraded_second_end_sec_ = 550.0;
  double local_elapsed_sec_ = 0.0;
  double bag_elapsed_sec_ = 0.0;
  bool in_fixed_degraded_window_ = false;
  std::string fixed_degraded_reason_ = "disabled";
  double local_fixed_degraded_start_sec_ = -1.0;
  double local_fixed_degraded_end_sec_ = -1.0;
  std::string local_mode_ = "NORMAL";
  std::string local_reinit_reason_ = "none";
  double local_mode_start_time_ = -1.0;
  V3D degraded_hold_entry_pos_ = V3D::Zero();
  StatesGroup degraded_hold_entry_state_;
  bool degraded_hold_entry_state_ready_ = false;
  V3D degraded_hold_entry_rpy_ = V3D::Zero();
  std::string degraded_hold_last_reject_reason_ = "none";
  bool local_map_cleared_last_ = false;
  bool visual_map_cleared_last_ = false;
  bool tracker_reset_last_ = false;
  int lio_bootstrap_frames_ = 0;
  int vio_bootstrap_frames_ = 0;
  int local_post_reinit_lio_frames_ = 30;
  int local_post_reinit_vio_frames_ = 90;
  double local_post_reinit_duration_sec_ = 5.0;
  double local_tracking_lost_window_sec_ = 1.0;
  int local_vio_unavailable_tracked_points_ = 2;
  double local_lio_weak_residual_threshold_ = 0.20;
  double local_vio_low_start_time_ = -1.0;
  double local_lio_weak_start_time_ = -1.0;
  bool local_vio_unavailable_ = false;
  bool local_lio_weak_ = false;

  bool corridor_prior_enable_ = false;
  bool corridor_prior_only_in_degenerate_ = true;
  double corridor_prior_axis_estimation_sec_ = 5.0;
  double corridor_prior_axis_estimation_max_sec_ = 10.0;
  double corridor_prior_min_axis_motion_ = 1.0;
  double corridor_prior_backward_window_sec_ = 5.0;
  double corridor_prior_backward_distance_threshold_ = 1.0;
  double corridor_prior_fail_safe_window_sec_ = 8.0;
  double corridor_prior_fail_safe_backward_distance_threshold_ = 2.0;
  std::string corridor_prior_backward_action_ = "downweight";
  double corridor_prior_lio_downweight_scale_ = 5.0;
  double corridor_prior_vio_downweight_scale_ = 10.0;
  bool corridor_prior_disable_map_update_on_downweight_ = true;
  bool corridor_prior_disable_visual_map_update_on_downweight_ = true;
  bool corridor_prior_started_ = false;
  bool corridor_prior_axis_ready_ = false;
  bool corridor_prior_axis_failed_ = false;
  double corridor_prior_entry_time_ = -1.0;
  V3D corridor_prior_entry_pos_ = V3D::Zero();
  V3D corridor_prior_axis_ = V3D::UnitX();
  std::deque<std::pair<double, double>> corridor_prior_progress_buffer_;
  double corridor_prior_progress_ = 0.0;
  double corridor_prior_progress_delta_1s_ = 0.0;
  double corridor_prior_progress_delta_window_ = 0.0;
  double corridor_prior_backward_distance_window_ = 0.0;
  double corridor_prior_backward_distance_fail_window_ = 0.0;
  std::string corridor_prior_action_ = "none";
  bool corridor_prior_update_voxel_map_enabled_ = true;
  bool corridor_prior_visual_map_update_enabled_ = true;

  ObservationQuality current_quality_;
  UpdateDecision current_decision_;
  bool update_decision_ready_ = false;
  std::string diagnostics_level_ = "summary";
  double diagnostics_summary_interval_sec_ = 1.0;
  double diagnostics_last_summary_time_ = -1.0;

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
  bool lio_force_voxel_map_update_ = true;
  double lio_force_map_update_interval_ = 0.3;
  int lio_force_map_update_lidar_frames_ = 3;
  int lio_frames_since_voxel_map_update_ = 0;
  double lio_last_voxel_map_update_time_ = -1.0;
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
  bool lio_freeze_state_when_degenerate_ = false;
  int lio_freeze_degenerate_min_frames_ = 1;
  int lio_degenerate_frame_count_ = 0;
  bool lio_state_jump_guard_en_ = true;
  double lio_state_jump_max_trans_m_ = 0.30;
  double lio_state_jump_max_rot_deg_ = 5.0;
  bool lio_freeze_state_ready_ = false;
  bool last_lio_stable_state_ready_ = false;
  StatesGroup lio_freeze_state_;
  StatesGroup last_lio_stable_state_;
  bool uwb_skip_when_lio_frozen_ = true;
  bool vio_visual_update_guard_en_ = true;
  double vio_visual_update_max_trans_m_ = 0.12;
  double vio_visual_update_max_rot_deg_ = 8.0;
  double vio_visual_update_max_trans_rate_mps_ = 3.0;
  double vio_visual_update_max_rot_rate_degps_ = 240.0;
  double vio_visual_update_max_backward_rate_mps_ = 0.5;
  double vio_visual_update_max_lateral_rate_mps_ = 1.0;
  double vio_visual_update_max_backward_m_ = 0.03;
  double vio_visual_update_max_backward_ratio_ = 0.08;
  double vio_visual_update_backward_abs_floor_m_ = 0.003;
  double vio_visual_update_max_lateral_m_ = 0.08;
  double vio_visual_update_max_lateral_ratio_ = 0.35;
  double vio_visual_update_max_exposure_delta_ = 0.30;
  std::string vio_visual_update_large_update_guard_action_ = "reject_update";
  std::string vio_visual_update_large_rotation_action_ = "downweight_update";
  bool vio_reject_visual_large_rotation_ = false;
  double vio_visual_update_large_rotation_noise_scale_ = 2.0;
  std::string vio_visual_update_backward_guard_action_ = "log_only";
  std::string vio_visual_update_lateral_guard_action_ = "log_only";
  std::string vio_visual_update_exposure_guard_action_ = "reject_update";
  std::string vio_visual_update_nonfinite_guard_action_ = "reject_update";
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

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points, degeneracy_guard_log_;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

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
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  ros::Publisher mavros_pose_publisher;
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
