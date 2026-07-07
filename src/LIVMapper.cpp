/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <system_error>
#include <unordered_set>
#include <unistd.h>

namespace fs = std::filesystem;

namespace
{
std::string formatLocalTime(const char *fmt)
{
  const std::time_t now = std::time(nullptr);
  std::tm tm_now;
  localtime_r(&now, &tm_now);
  char buf[64] = {0};
  std::strftime(buf, sizeof(buf), fmt, &tm_now);
  return std::string(buf);
}

std::string currentTimeForFilename()
{
  return formatLocalTime("%Y-%m-%d-%H%M%S");
}

std::string ensureTrailingSlash(std::string path)
{
  if (!path.empty() && path.back() != '/') path += '/';
  return path;
}

std::string makeRunOutputDir(const std::string &base_dir)
{
  const std::string normalized_base = ensureTrailingSlash(base_dir);
  const std::string run_dir = normalized_base + currentTimeForFilename();
  std::error_code ec;
  fs::create_directories(run_dir, ec);
  if (ec)
  {
    std::cerr << "[ LIVMapper ] Warning: failed to create output dir: "
              << run_dir << " (" << ec.message() << ")" << std::endl;
    return normalized_base;
  }
  return ensureTrailingSlash(run_dir);
}
}

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  uwb_manager.reset(new UwbManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  uwb_manager->initialize(nh, save_path);
  initializeComponents(nh);
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper()
{
  if (uwb_manager) uwb_manager->shutdown();
  if (udp_socket_fd_ >= 0)
  {
    ::close(udp_socket_fd_);
    udp_socket_fd_ = -1;
  }
}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<bool>("experimental_features/enable", experimental_features_enable_, false);
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");
  nh.param<int>("common/sub_lidar_queue_size", sub_lidar_queue_size_, 128);
  nh.param<int>("common/sub_imu_queue_size", sub_imu_queue_size_, 512);
  nh.param<int>("common/sub_img_queue_size", sub_img_queue_size_, 16);
  nh.param<int>("common/max_lidar_buffer_size", max_lidar_buffer_size_, 32);
  nh.param<int>("common/max_imu_buffer_size", max_imu_buffer_size_, 3000);
  nh.param<int>("common/max_img_buffer_size", max_img_buffer_size_, 12);
  nh.param<int>("common/max_prop_imu_buffer_size", max_prop_imu_buffer_size_, 3000);
  nh.param<int>("common/sync_img_buffer_min_size", sync_img_buffer_min_size_, 1);
  nh.param<double>("common/sync_img_lookahead_time", sync_img_lookahead_time_, 0.0);
  sync_img_buffer_min_size_ = std::max(1, sync_img_buffer_min_size_);
  sync_img_lookahead_time_ = std::max(0.0, sync_img_lookahead_time_);

  bool legacy_visual_update_serial = true;
  nh.param<bool>("vio/deterministic_visual_update_en", legacy_visual_update_serial, true);
  nh.param<bool>("deterministic_debug/state_snap_en", deterministic_state_snap_en_, true);
  nh.param<bool>("deterministic_debug/pixel_snap_en", deterministic_pixel_snap_en_, true);
  nh.param<bool>("deterministic_debug/camera_point_snap_en", deterministic_camera_point_snap_en_, true);
  nh.param<bool>("deterministic_debug/contiguous_image_copy_en", deterministic_contiguous_image_copy_en_, true);
  nh.param<bool>("deterministic_debug/visual_update_serial_en", vio_deterministic_visual_update_en_, legacy_visual_update_serial);
  nh.param<bool>("deterministic_debug/imu_accept_out_of_order_en", deterministic_imu_accept_out_of_order_en_, true);
  nh.param<bool>("deterministic_debug/imu_buffer_sort_en", deterministic_imu_buffer_sort_en_, true);
  nh.param<bool>("deterministic_debug/prop_imu_buffer_sort_en", deterministic_prop_imu_buffer_sort_en_, true);
  nh.param<bool>("deterministic_debug/image_buffer_sort_en", deterministic_image_buffer_sort_en_, true);
  nh.param<bool>("deterministic_debug/sync_wait_for_image_lookahead_en", deterministic_sync_wait_for_image_lookahead_en_, true);
  nh.param<bool>("deterministic_debug/pending_vio_image_en", deterministic_pending_vio_image_en_, true);
  nh.param<bool>("deterministic_debug/lio_feature_sort_en", deterministic_lio_feature_sort_en_, true);
  nh.param<bool>("deterministic_debug/visual_observed_voxel_sort_en", deterministic_visual_observed_voxel_sort_en_, true);
  nh.param<bool>("deterministic_debug/visual_voxel_key_sort_en", deterministic_visual_voxel_key_sort_en_, true);
  setStateSnapForDeterminismEnabled(deterministic_state_snap_en_);

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  IMG_POINT_COV = experimental_features_enable_ ? 200.0 : 100.0;
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<bool>("vio/visual_map_prune_en", visual_map_prune_en, true);
  nh.param<int>("vio/visual_map_max_voxels", visual_map_max_voxels, 1800);
  nh.param<int>("vio/visual_map_max_points_per_voxel", visual_map_max_points_per_voxel, 10);
  nh.param<int>("vio/visual_map_max_total_points", visual_map_max_total_points, 20000);
  nh.param<int>("vio/visual_map_max_add_per_frame", visual_map_max_add_per_frame_, 300);
  nh.param<double>("vio/visual_map_min_shi_tomasi_score", visual_map_min_shi_tomasi_score_, 10.0);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);
  vio_min_retrieve_points_ = 45;
  vio_min_update_meas_ = 900;
  vio_low_track_force_update_stride_ = 0;
  vio_low_track_force_min_points_ = 8;
  vio_max_state_update_rot_deg_ = 0.8;
  vio_max_state_update_trans_m_ = 0.08;
  nh.param<int>("vio/min_retrieve_points", vio_min_retrieve_points_, 45);
  nh.param<int>("vio/min_update_meas", vio_min_update_meas_, 900);
  nh.param<int>("vio/low_track_force_update_stride", vio_low_track_force_update_stride_, 0);
  nh.param<int>("vio/low_track_force_min_points", vio_low_track_force_min_points_, 8);
  nh.param<bool>("vio/visual_update_guard_en", vio_visual_update_guard_en_, true);
  nh.param<double>("vio/visual_update_max_trans_m", vio_visual_update_max_trans_m_, 0.12);
  nh.param<double>("vio/visual_update_max_rot_deg", vio_visual_update_max_rot_deg_, 8.0);
  nh.param<double>("vio/visual_update_max_trans_rate_mps", vio_visual_update_max_trans_rate_mps_, 3.0);
  nh.param<double>("vio/visual_update_max_rot_rate_degps", vio_visual_update_max_rot_rate_degps_, 240.0);
  nh.param<double>("vio/visual_update_max_backward_rate_mps", vio_visual_update_max_backward_rate_mps_, 0.5);
  nh.param<double>("vio/visual_update_max_lateral_rate_mps", vio_visual_update_max_lateral_rate_mps_, 1.0);
  nh.param<double>("vio/visual_update_max_backward_m", vio_visual_update_max_backward_m_, 0.03);
  nh.param<double>("vio/visual_update_max_backward_ratio", vio_visual_update_max_backward_ratio_, 0.08);
  nh.param<double>("vio/visual_update_backward_abs_floor_m", vio_visual_update_backward_abs_floor_m_, 0.003);
  nh.param<double>("vio/visual_update_max_lateral_m", vio_visual_update_max_lateral_m_, 0.08);
  nh.param<double>("vio/visual_update_max_lateral_ratio", vio_visual_update_max_lateral_ratio_, 0.35);
  nh.param<double>("vio/visual_update_max_exposure_delta", vio_visual_update_max_exposure_delta_, 0.30);
  nh.param<std::string>("vio/visual_update_large_update_guard_action", vio_visual_update_large_update_guard_action_, "reject_update");
  nh.param<std::string>("vio/visual_update_large_rotation_action", vio_visual_update_large_rotation_action_, "downweight_update");
  nh.param<bool>("vio/reject_visual_large_rotation", vio_reject_visual_large_rotation_, false);
  nh.param<double>("vio/visual_update_large_rotation_noise_scale", vio_visual_update_large_rotation_noise_scale_, 2.0);
  nh.param<std::string>("vio/visual_update_backward_guard_action", vio_visual_update_backward_guard_action_, "log_only");
  nh.param<std::string>("vio/visual_update_lateral_guard_action", vio_visual_update_lateral_guard_action_, "log_only");
  nh.param<std::string>("vio/visual_update_exposure_guard_action", vio_visual_update_exposure_guard_action_, "reject_update");
  nh.param<std::string>("vio/visual_update_nonfinite_guard_action", vio_visual_update_nonfinite_guard_action_, "reject_update");
  nh.param<bool>("vio/image_quality_gate_en", vio_image_quality_gate_en_, false);
  nh.param<double>("vio/image_quality_max_saturated_fraction", vio_image_quality_max_saturated_fraction_, 0.20);
  nh.param<double>("vio/image_quality_max_tile_saturated_fraction", vio_image_quality_max_tile_saturated_fraction_, 0.35);
  nh.param<double>("vio/image_quality_max_dark_fraction", vio_image_quality_max_dark_fraction_, 0.98);
  nh.param<double>("vio/image_quality_min_intensity_std", vio_image_quality_min_intensity_std_, 6.0);
  nh.param<bool>("vio/visual_patch_quality_gate_en", vio_visual_patch_quality_gate_en_, true);
  nh.param<double>("vio/visual_patch_max_saturated_fraction", vio_visual_patch_max_saturated_fraction_, 0.10);
  nh.param<double>("vio/visual_patch_min_intensity_std", vio_visual_patch_min_intensity_std_, 2.0);
  nh.param<int>("vio/image_quality_saturated_pixel_value", vio_image_quality_saturated_pixel_value_, 250);
  nh.param<int>("vio/image_quality_dark_pixel_value", vio_image_quality_dark_pixel_value_, 5);
  nh.param<int>("vio/image_quality_tile_rows", vio_image_quality_tile_rows_, 4);
  nh.param<int>("vio/image_quality_tile_cols", vio_image_quality_tile_cols_, 4);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);
  nh.param<bool>("uav/lock_z_after_gravity_align_en", legacy_lock_z_after_gravity_align_en_, false);
  if (legacy_lock_z_after_gravity_align_en_)
  {
    ROS_WARN("[DEGEN_GUARD] uav/lock_z_after_gravity_align_en is deprecated and ignored; use degeneracy_guard/enable_z_soft_constraint.");
  }
  nh.param<bool>("degeneracy_guard/enable", deg_guard_enable_, false);
  nh.param<bool>("degeneracy_guard/enable_z_soft_constraint", deg_guard_enable_z_soft_constraint_, true);
  nh.param<double>("degeneracy_guard/z_ref", deg_guard_z_ref_, 0.0);
  nh.param<double>("degeneracy_guard/sigma_z", deg_guard_sigma_z_, 0.20);
  nh.param<double>("degeneracy_guard/sigma_vz", deg_guard_sigma_vz_, 0.30);
  nh.param<double>("degeneracy_guard/z_soft_gain", deg_guard_z_gain_, 1.0);
  nh.param<bool>("degeneracy_guard/enable_nhc", deg_guard_enable_nhc_, true);
  nh.param<double>("degeneracy_guard/sigma_body_vy", deg_guard_sigma_body_vy_, 0.05);
  nh.param<double>("degeneracy_guard/sigma_body_vz", deg_guard_sigma_body_vz_, 0.05);
  nh.param<double>("degeneracy_guard/nhc_min_speed", deg_guard_nhc_min_speed_, 0.05);
  nh.param<bool>("degeneracy_guard/nhc_only_in_degenerate", deg_guard_nhc_only_in_degenerate_, true);
  nh.param<double>("degeneracy_guard/nhc_gain", deg_guard_nhc_gain_, 1.0);
  nh.param<bool>("degeneracy_guard/enable_backward_guard", deg_guard_enable_backward_guard_, true);
  nh.param<double>("degeneracy_guard/backward_step_threshold", deg_guard_backward_step_threshold_, 0.05);
  nh.param<double>("degeneracy_guard/backward_speed_threshold", deg_guard_backward_speed_threshold_, 0.20);
  nh.param<int>("degeneracy_guard/backward_consecutive_frames", deg_guard_backward_consecutive_frames_, 10);
  nh.param<std::string>("degeneracy_guard/backward_guard_action", deg_guard_backward_action_, "log_only");
  nh.param<bool>("degeneracy_guard/enable_corridor_detection", deg_guard_enable_corridor_detection_, true);
  nh.param<int>("degeneracy_guard/min_lidar_features", deg_guard_min_lidar_features_, 30);
  nh.param<int>("degeneracy_guard/min_visual_tracked_points", deg_guard_min_visual_tracked_points_, 2);
  nh.param<int>("degeneracy_guard/vio_low_feature_tracked_points", deg_guard_vio_low_feature_tracked_points_, 5);
  nh.param<bool>("degeneracy_guard/use_vio_skip_for_degenerate", deg_guard_use_vio_skip_for_degenerate_, false);
  nh.param<bool>("degeneracy_guard/use_vio_large_rotation_for_reject", deg_guard_use_vio_large_rotation_for_reject_, false);
  nh.param<bool>("degeneracy_guard/reject_visual_large_rotation", vio_reject_visual_large_rotation_, vio_reject_visual_large_rotation_);
  nh.param<double>("degeneracy_guard/camera_dt", deg_guard_camera_dt_, 1.0 / 30.0);
  nh.param<double>("degeneracy_guard/lidar_dt", deg_guard_lidar_dt_, 0.1);
  nh.param<double>("degeneracy_guard/max_update_translation_norm", deg_guard_max_update_translation_norm_, 0.5);
  nh.param<double>("degeneracy_guard/max_update_yaw_deg", deg_guard_max_update_yaw_deg_, 5.0);
  nh.param<double>("degeneracy_guard/max_update_translation_rate_mps", deg_guard_max_update_translation_rate_mps_, 3.0);
  nh.param<double>("degeneracy_guard/max_update_yaw_rate_degps", deg_guard_max_update_yaw_rate_degps_, 180.0);
  nh.param<double>("degeneracy_guard/hessian_condition_threshold", deg_guard_hessian_condition_threshold_, 1000.0);
  nh.param<int>("degeneracy_guard/min_degenerate_frames", deg_guard_min_degenerate_frames_, 3);
  nh.param<int>("degeneracy_guard/recover_frames", deg_guard_recover_frames_, 3);
  nh.param<double>("degeneracy_guard/degenerate_lio_noise_scale", deg_guard_degenerate_lio_noise_scale_, 1.2);
  nh.param<double>("degeneracy_guard/degenerate_vio_noise_scale", deg_guard_degenerate_vio_noise_scale_, 2.0);
  nh.param<bool>("degeneracy_guard/enable_adaptive_sensor_weighting", deg_guard_enable_adaptive_sensor_weighting_, true);
  nh.param<double>("degeneracy_guard/adaptive_lio_base_noise_scale", deg_guard_adaptive_lio_base_noise_scale_, 1.0);
  nh.param<double>("degeneracy_guard/adaptive_vio_base_noise_scale", deg_guard_adaptive_vio_base_noise_scale_, 1.0);
  nh.param<double>("degeneracy_guard/adaptive_lio_low_feature_noise_scale", deg_guard_adaptive_lio_low_feature_noise_scale_, 2.0);
  nh.param<double>("degeneracy_guard/adaptive_vio_low_track_noise_scale", deg_guard_adaptive_vio_low_track_noise_scale_, 20.0);
  nh.param<double>("degeneracy_guard/vio_low_feature_noise_scale", deg_guard_adaptive_vio_low_track_noise_scale_, deg_guard_adaptive_vio_low_track_noise_scale_);
  nh.param<double>("degeneracy_guard/adaptive_lio_high_residual_noise_scale", deg_guard_adaptive_lio_high_residual_noise_scale_, 2.0);
  nh.param<double>("degeneracy_guard/adaptive_lio_residual_ref", deg_guard_adaptive_lio_residual_ref_, 0.05);
  nh.param<double>("degeneracy_guard/adaptive_max_noise_scale", deg_guard_adaptive_max_noise_scale_, 10.0);
  nh.param<int>("degeneracy_guard/vio_skip_tracked_points", deg_guard_vio_skip_tracked_points_, 1);
  nh.param<int>("degeneracy_guard/vio_skip_min_tracked_points", deg_guard_vio_skip_min_tracked_points_, deg_guard_vio_skip_tracked_points_ + 1);
  nh.param<int>("degeneracy_guard/vio_min_update_meas", deg_guard_vio_min_update_meas_, 32);
  nh.param<bool>("degeneracy_guard/reject_large_update_in_degenerate", deg_guard_reject_large_update_in_degenerate_, false);
  nh.param<bool>("degeneracy_guard/reject_nonfinite_update", deg_guard_reject_nonfinite_update_, true);
	  nh.param<double>("degeneracy_guard/max_degenerate_update_translation", deg_guard_max_degenerate_update_translation_, 0.3);
	  nh.param<double>("degeneracy_guard/max_degenerate_update_yaw_deg", deg_guard_max_degenerate_update_yaw_deg_, 3.0);
	  nh.param<std::string>("degeneracy_guard/log_file", deg_guard_log_file_, "/tmp/degeneracy_guard_log.txt");
	  nh.param<bool>("deterministic_mode", deterministic_mode_, true);
	  nh.param<bool>("deterministic/mode", deterministic_mode_, deterministic_mode_);
	  nh.param<double>("uwb/update_window_sec", uwb_update_window_sec_, 0.05);
	  nh.param<bool>("uwb/relocalize_en", uwb_relocalize_en_, false);
	  nh.param<double>("uwb/relocalize_xy_threshold", uwb_relocalize_xy_threshold_, 1.0);
	  nh.param<bool>("uwb/update_only_on_lio", uwb_update_only_on_lio_, true);
	  nh.param<bool>("safety_guard/enable_safety_guard", safety_guard_enable_, false);
  nh.param<bool>("safety_guard/enable", safety_guard_enable_, safety_guard_enable_);
  nh.param<double>("safety_guard/max_speed", safety_max_speed_, 3.0);
  nh.param<double>("safety_guard/max_frame_translation", safety_max_frame_translation_, 0.5);
  nh.param<double>("safety_guard/max_frame_rotation_deg", safety_max_frame_rotation_deg_, 15.0);
  nh.param<int>("safety_guard/fail_safe_recover_frames", safety_fail_safe_recover_frames_, 10);
  nh.param<double>("safety_guard/backward_time_window", safety_backward_time_window_, 5.0);
  nh.param<double>("safety_guard/backward_distance_threshold", safety_backward_distance_threshold_, 1.0);
  nh.param<std::string>("safety_guard/backward_action", safety_backward_action_, "log_only");
  nh.param<bool>("corridor_motion_prior/enable", corridor_prior_enable_, false);
  nh.param<bool>("corridor_motion_prior/only_in_degenerate", corridor_prior_only_in_degenerate_, true);
  nh.param<double>("corridor_motion_prior/axis_estimation_sec", corridor_prior_axis_estimation_sec_, 5.0);
  nh.param<double>("corridor_motion_prior/axis_estimation_max_sec", corridor_prior_axis_estimation_max_sec_, 10.0);
  nh.param<double>("corridor_motion_prior/min_axis_motion", corridor_prior_min_axis_motion_, 1.0);
  nh.param<double>("corridor_motion_prior/backward_window_sec", corridor_prior_backward_window_sec_, 5.0);
  nh.param<double>("corridor_motion_prior/backward_distance_threshold", corridor_prior_backward_distance_threshold_, 1.0);
  nh.param<double>("corridor_motion_prior/fail_safe_window_sec", corridor_prior_fail_safe_window_sec_, 8.0);
  nh.param<double>("corridor_motion_prior/fail_safe_backward_distance_threshold", corridor_prior_fail_safe_backward_distance_threshold_, 2.0);
  nh.param<std::string>("corridor_motion_prior/backward_action", corridor_prior_backward_action_, "downweight");
  nh.param<double>("corridor_motion_prior/lio_downweight_scale", corridor_prior_lio_downweight_scale_, 5.0);
  nh.param<double>("corridor_motion_prior/vio_downweight_scale", corridor_prior_vio_downweight_scale_, 10.0);
  nh.param<bool>("corridor_motion_prior/disable_map_update_on_downweight", corridor_prior_disable_map_update_on_downweight_, true);
  nh.param<bool>("corridor_motion_prior/disable_visual_map_update_on_downweight", corridor_prior_disable_visual_map_update_on_downweight_, true);
  deg_guard_sigma_z_ = std::max(1e-6, deg_guard_sigma_z_);
  deg_guard_sigma_vz_ = std::max(1e-6, deg_guard_sigma_vz_);
  deg_guard_sigma_body_vy_ = std::max(1e-6, deg_guard_sigma_body_vy_);
  deg_guard_sigma_body_vz_ = std::max(1e-6, deg_guard_sigma_body_vz_);
  deg_guard_z_gain_ = std::max(0.0, std::min(1.0, deg_guard_z_gain_));
  deg_guard_nhc_gain_ = std::max(0.0, std::min(1.0, deg_guard_nhc_gain_));
  deg_guard_nhc_min_speed_ = std::max(0.0, deg_guard_nhc_min_speed_);
  deg_guard_backward_step_threshold_ = std::max(0.0, deg_guard_backward_step_threshold_);
  deg_guard_backward_speed_threshold_ = std::max(0.0, deg_guard_backward_speed_threshold_);
  deg_guard_backward_consecutive_frames_ = std::max(1, deg_guard_backward_consecutive_frames_);
  deg_guard_min_lidar_features_ = std::max(0, deg_guard_min_lidar_features_);
  deg_guard_min_visual_tracked_points_ = std::max(0, deg_guard_min_visual_tracked_points_);
  deg_guard_vio_low_feature_tracked_points_ = std::max(0, deg_guard_vio_low_feature_tracked_points_);
  deg_guard_camera_dt_ = std::max(1e-4, deg_guard_camera_dt_);
  deg_guard_lidar_dt_ = std::max(1e-4, deg_guard_lidar_dt_);
  deg_guard_max_update_translation_norm_ = std::max(0.0, deg_guard_max_update_translation_norm_);
  deg_guard_max_update_yaw_deg_ = std::max(0.0, deg_guard_max_update_yaw_deg_);
  deg_guard_max_update_translation_rate_mps_ = std::max(0.0, deg_guard_max_update_translation_rate_mps_);
  deg_guard_max_update_yaw_rate_degps_ = std::max(0.0, deg_guard_max_update_yaw_rate_degps_);
  deg_guard_hessian_condition_threshold_ = std::max(1.0, deg_guard_hessian_condition_threshold_);
  deg_guard_min_degenerate_frames_ = std::max(1, deg_guard_min_degenerate_frames_);
  deg_guard_recover_frames_ = std::max(1, deg_guard_recover_frames_);
  deg_guard_degenerate_lio_noise_scale_ = std::max(1.0, deg_guard_degenerate_lio_noise_scale_);
  deg_guard_degenerate_vio_noise_scale_ = std::max(1.0, deg_guard_degenerate_vio_noise_scale_);
  deg_guard_adaptive_lio_base_noise_scale_ = std::max(1.0, deg_guard_adaptive_lio_base_noise_scale_);
  deg_guard_adaptive_vio_base_noise_scale_ = std::max(1.0, deg_guard_adaptive_vio_base_noise_scale_);
  deg_guard_adaptive_lio_low_feature_noise_scale_ = std::max(1.0, deg_guard_adaptive_lio_low_feature_noise_scale_);
  deg_guard_adaptive_vio_low_track_noise_scale_ = std::max(1.0, deg_guard_adaptive_vio_low_track_noise_scale_);
  deg_guard_adaptive_lio_high_residual_noise_scale_ = std::max(1.0, deg_guard_adaptive_lio_high_residual_noise_scale_);
  deg_guard_adaptive_lio_residual_ref_ = std::max(1e-6, deg_guard_adaptive_lio_residual_ref_);
  deg_guard_adaptive_max_noise_scale_ = std::max(1.0, deg_guard_adaptive_max_noise_scale_);
  deg_guard_max_degenerate_update_translation_ = std::max(0.0, deg_guard_max_degenerate_update_translation_);
  deg_guard_max_degenerate_update_yaw_deg_ = std::max(0.0, deg_guard_max_degenerate_update_yaw_deg_);
  safety_max_speed_ = std::max(0.0, safety_max_speed_);
  safety_max_frame_translation_ = std::max(0.0, safety_max_frame_translation_);
  safety_max_frame_rotation_deg_ = std::max(0.0, safety_max_frame_rotation_deg_);
  safety_fail_safe_recover_frames_ = std::max(1, safety_fail_safe_recover_frames_);
  safety_backward_time_window_ = std::max(0.1, safety_backward_time_window_);
  safety_backward_distance_threshold_ = std::max(0.0, safety_backward_distance_threshold_);
  corridor_prior_axis_estimation_sec_ = std::max(0.1, corridor_prior_axis_estimation_sec_);
  corridor_prior_axis_estimation_max_sec_ =
      std::max(corridor_prior_axis_estimation_sec_, corridor_prior_axis_estimation_max_sec_);
  corridor_prior_min_axis_motion_ = std::max(0.0, corridor_prior_min_axis_motion_);
  corridor_prior_backward_window_sec_ = std::max(0.1, corridor_prior_backward_window_sec_);
  corridor_prior_backward_distance_threshold_ = std::max(0.0, corridor_prior_backward_distance_threshold_);
  corridor_prior_fail_safe_window_sec_ =
      std::max(corridor_prior_backward_window_sec_, corridor_prior_fail_safe_window_sec_);
  corridor_prior_fail_safe_backward_distance_threshold_ =
      std::max(corridor_prior_backward_distance_threshold_, corridor_prior_fail_safe_backward_distance_threshold_);
	  corridor_prior_lio_downweight_scale_ = std::max(1.0, corridor_prior_lio_downweight_scale_);
	  corridor_prior_vio_downweight_scale_ = std::max(1.0, corridor_prior_vio_downweight_scale_);
	  uwb_update_window_sec_ = std::max(0.0, uwb_update_window_sec_);
	  uwb_relocalize_xy_threshold_ = std::max(0.0, uwb_relocalize_xy_threshold_);
	  auto normalizeGuardAction = [](std::string &action, const std::string &fallback) {
    if (action == "downweight") action = "downweight_update";
    if (action == "none" || action == "log_only" || action == "downweight_update" ||
        action == "reject_update" || action == "fail_safe_or_downweight") return;
    ROS_WARN("[DEGEN_GUARD] Unknown guard action=%s, fallback to %s.", action.c_str(), fallback.c_str());
    action = fallback;
  };
  normalizeGuardAction(deg_guard_backward_action_, "log_only");
  if (deg_guard_backward_action_ == "reject_update")
  {
    ROS_WARN("[DEGEN_GUARD] backward_guard_action=reject_update is deprecated; use corridor_motion_prior window decision instead.");
    deg_guard_backward_action_ = "log_only";
  }
  normalizeGuardAction(vio_visual_update_large_update_guard_action_, "reject_update");
  normalizeGuardAction(vio_visual_update_large_rotation_action_, "downweight_update");
  normalizeGuardAction(vio_visual_update_backward_guard_action_, "log_only");
  normalizeGuardAction(vio_visual_update_lateral_guard_action_, "log_only");
  normalizeGuardAction(vio_visual_update_exposure_guard_action_, "reject_update");
  normalizeGuardAction(vio_visual_update_nonfinite_guard_action_, "reject_update");
  normalizeGuardAction(safety_backward_action_, "log_only");
  if (corridor_prior_backward_action_ == "downweight_update") corridor_prior_backward_action_ = "downweight";
  if (corridor_prior_backward_action_ != "none" &&
      corridor_prior_backward_action_ != "log_only" &&
      corridor_prior_backward_action_ != "downweight" &&
      corridor_prior_backward_action_ != "fail_safe")
  {
    ROS_WARN("[CORRIDOR_PRIOR] Unknown backward_action=%s, fallback to downweight.",
             corridor_prior_backward_action_.c_str());
    corridor_prior_backward_action_ = "downweight";
  }
  deg_guard_vio_skip_tracked_points_ = std::max(0, deg_guard_vio_skip_tracked_points_);
  deg_guard_vio_skip_min_tracked_points_ = std::max(std::max(0, deg_guard_vio_skip_min_tracked_points_),
                                                     deg_guard_vio_skip_tracked_points_ + 1);
  deg_guard_vio_min_update_meas_ = std::max(1, deg_guard_vio_min_update_meas_);
  vio_visual_update_max_trans_rate_mps_ = std::max(0.0, vio_visual_update_max_trans_rate_mps_);
  vio_visual_update_max_rot_rate_degps_ = std::max(0.0, vio_visual_update_max_rot_rate_degps_);
  vio_visual_update_max_backward_rate_mps_ = std::max(0.0, vio_visual_update_max_backward_rate_mps_);
  vio_visual_update_max_lateral_rate_mps_ = std::max(0.0, vio_visual_update_max_lateral_rate_mps_);
  vio_visual_update_large_rotation_noise_scale_ = std::max(1.0, vio_visual_update_large_rotation_noise_scale_);
  nh.param<bool>("uwb/debug_output_correction_en", uwb_output_correction_en_, false);
  nh.param<bool>("uwb/output_correction_en", uwb_output_correction_en_, uwb_output_correction_en_);
  nh.param<bool>("uwb/output_smooth_en", uwb_output_smooth_en_, true);
  nh.param<double>("uwb/output_smooth_alpha", uwb_output_smooth_alpha_, 0.15);
  nh.param<double>("uwb/output_smooth_max_step_m", uwb_output_smooth_max_step_m_, 0.05);
	  nh.param<bool>("uwb/skip_when_lio_frozen", uwb_skip_when_lio_frozen_, true);
	  uwb_output_smooth_alpha_ = std::max(0.0, std::min(1.0, uwb_output_smooth_alpha_));
	  uwb_output_smooth_max_step_m_ = std::max(0.0, uwb_output_smooth_max_step_m_);
			  nh.param<bool>("local_reinit/enable", local_reinit_enable_, true);
		  nh.param<bool>("debug_fixed_degraded_intervals/enable", debug_fixed_degraded_intervals_enable_, false);
		  // Compatibility only: old elevator_mode is now debug-only and must be explicitly enabled.
		  nh.param<bool>("elevator_mode/enable", debug_fixed_degraded_intervals_enable_, debug_fixed_degraded_intervals_enable_);
			  nh.param<std::string>("debug_fixed_degraded_intervals/trigger_mode", fixed_degraded_trigger_mode_, "manual_time");
			  nh.param<std::string>("elevator_mode/trigger_mode", fixed_degraded_trigger_mode_, fixed_degraded_trigger_mode_);
			  nh.param<bool>("degraded_bootstrap/enable", degraded_bootstrap_enable_, true);
			  nh.param<bool>("post_elevator_reinit/enable", degraded_bootstrap_enable_, degraded_bootstrap_enable_);
		  nh.param<bool>("degraded_hold/disable_visual_map", disable_visual_map_in_degraded_hold_, true);
		  nh.param<bool>("degraded_hold/disable_voxel_map", disable_voxel_map_in_degraded_hold_, true);
		  nh.param<bool>("elevator_mode/disable_visual_map", disable_visual_map_in_degraded_hold_, disable_visual_map_in_degraded_hold_);
		  nh.param<bool>("elevator_mode/disable_voxel_map", disable_voxel_map_in_degraded_hold_, disable_voxel_map_in_degraded_hold_);
			  nh.param<double>("degraded_hold/attitude_reject_deg", degraded_hold_attitude_reject_deg_, 5.0);
			  nh.param<double>("degraded_hold/speed_reject_mps", degraded_hold_speed_reject_mps_, 1.0);
			  nh.param<double>("elevator_mode/attitude_reject_deg", degraded_hold_attitude_reject_deg_, degraded_hold_attitude_reject_deg_);
			  nh.param<double>("elevator_mode/speed_reject_mps", degraded_hold_speed_reject_mps_, degraded_hold_speed_reject_mps_);
		  nh.param<double>("debug_fixed_degraded_intervals/bag_start_offset", bag_start_offset_, 0.0);
		  nh.param<double>("elevator_mode/bag_start_offset", bag_start_offset_, bag_start_offset_);
		  nh.param<double>("bag_start_offset", bag_start_offset_, bag_start_offset_);
		  nh.param<double>("debug_fixed_degraded_intervals/first_start_sec", fixed_degraded_first_start_sec_, 150.0);
		  nh.param<double>("debug_fixed_degraded_intervals/first_end_sec", fixed_degraded_first_end_sec_, 205.0);
		  nh.param<double>("debug_fixed_degraded_intervals/second_start_sec", fixed_degraded_second_start_sec_, 525.0);
		  nh.param<double>("debug_fixed_degraded_intervals/second_end_sec", fixed_degraded_second_end_sec_, 565.0);
		  nh.param<double>("elevator_mode/first_start_sec", fixed_degraded_first_start_sec_, fixed_degraded_first_start_sec_);
		  nh.param<double>("elevator_mode/first_end_sec", fixed_degraded_first_end_sec_, fixed_degraded_first_end_sec_);
		  nh.param<double>("elevator_mode/second_start_sec", fixed_degraded_second_start_sec_, fixed_degraded_second_start_sec_);
		  nh.param<double>("elevator_mode/second_end_sec", fixed_degraded_second_end_sec_, fixed_degraded_second_end_sec_);
		  nh.param<double>("debug_fixed_degraded_intervals/first_start_sec", local_fixed_degraded_start_sec_, fixed_degraded_first_start_sec_);
		  nh.param<double>("debug_fixed_degraded_intervals/first_end_sec", local_fixed_degraded_end_sec_, fixed_degraded_first_end_sec_);
		  fixed_degraded_first_start_sec_ = local_fixed_degraded_start_sec_;
		  fixed_degraded_first_end_sec_ = local_fixed_degraded_end_sec_;
		  nh.param<std::string>("diagnostics/level", diagnostics_level_, "summary");
		  nh.param<double>("diagnostics/summary_interval_sec", diagnostics_summary_interval_sec_, 1.0);
		  nh.param<int>("local_reinit/post_lio_bootstrap_frames", local_post_reinit_lio_frames_, 30);
	  nh.param<int>("local_reinit/post_vio_bootstrap_frames", local_post_reinit_vio_frames_, 90);
	  nh.param<double>("local_reinit/post_duration_sec", local_post_reinit_duration_sec_, 5.0);
	  nh.param<double>("local_reinit/tracking_lost_window_sec", local_tracking_lost_window_sec_, 1.0);
	  nh.param<int>("local_reinit/vio_unavailable_tracked_points", local_vio_unavailable_tracked_points_, 2);
	  nh.param<double>("local_reinit/lio_weak_residual_threshold", local_lio_weak_residual_threshold_, 0.20);
	  local_post_reinit_lio_frames_ = std::max(1, local_post_reinit_lio_frames_);
	  local_post_reinit_vio_frames_ = std::max(1, local_post_reinit_vio_frames_);
	  local_post_reinit_duration_sec_ = std::max(0.0, local_post_reinit_duration_sec_);
		  local_tracking_lost_window_sec_ = std::max(0.1, local_tracking_lost_window_sec_);
		  local_vio_unavailable_tracked_points_ = std::max(0, local_vio_unavailable_tracked_points_);
			  local_lio_weak_residual_threshold_ = std::max(0.0, local_lio_weak_residual_threshold_);
			  degraded_hold_attitude_reject_deg_ = std::max(0.0, degraded_hold_attitude_reject_deg_);
			  degraded_hold_speed_reject_mps_ = std::max(0.0, degraded_hold_speed_reject_mps_);
			  bag_start_offset_ = std::max(0.0, bag_start_offset_);
		  fixed_degraded_first_start_sec_ = std::max(0.0, fixed_degraded_first_start_sec_);
		  fixed_degraded_first_end_sec_ = std::max(fixed_degraded_first_start_sec_, fixed_degraded_first_end_sec_);
		  fixed_degraded_second_start_sec_ = std::max(0.0, fixed_degraded_second_start_sec_);
		  fixed_degraded_second_end_sec_ = std::max(fixed_degraded_second_start_sec_, fixed_degraded_second_end_sec_);
		  if (diagnostics_level_ != "off" && diagnostics_level_ != "summary" && diagnostics_level_ != "verbose")
		  {
		    ROS_WARN("[Diagnostics] Unknown diagnostics.level=%s, fallback to summary.", diagnostics_level_.c_str());
		    diagnostics_level_ = "summary";
		  }
		  diagnostics_summary_interval_sec_ = std::max(0.1, diagnostics_summary_interval_sec_);
	  nh.param<bool>("lio/freeze_state_when_degenerate", lio_freeze_state_when_degenerate_, false);
  nh.param<int>("lio/freeze_degenerate_min_frames", lio_freeze_degenerate_min_frames_, 1);
  nh.param<bool>("lio/state_jump_guard_en", lio_state_jump_guard_en_, true);
  nh.param<double>("lio/state_jump_max_trans_m", lio_state_jump_max_trans_m_, 0.30);
  nh.param<double>("lio/state_jump_max_rot_deg", lio_state_jump_max_rot_deg_, 5.0);
  lio_freeze_degenerate_min_frames_ = std::max(1, lio_freeze_degenerate_min_frames_);
  lio_state_jump_max_trans_m_ = std::max(0.0, lio_state_jump_max_trans_m_);
  lio_state_jump_max_rot_deg_ = std::max(0.0, lio_state_jump_max_rot_deg_);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<string>("/laserMapping/save_path",save_path,""); 
  if (save_path.empty()) nh.param<string>("pcd_save/save_path",save_path,"/home/jetson/data");
  save_path = makeRunOutputDir(save_path);
  nh.param<bool>("pcd_save/global_map_pub", global_map_pub, false);  // 新增：读取 global_map_pub 参数，默认 false
  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<bool>("pcd_save/save_log_en", save_log_en, true);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<int>("pcd_save/max_cache_points", pcd_cache_max_points, 300000);
  nh.param<bool>("udp_report/en", udp_report_en, false);
  nh.param<std::string>("udp_report/target_ip", udp_target_ip_, "127.0.0.1");
  nh.param<int>("udp_report/target_port", udp_report_port_, 9000);
  nh.param<std::string>("udp_report/device_id", udp_device_id_, "fast_livo2");
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);
  nh.param<bool>("publish/colorize_cloud_en", colorize_cloud_en_, true);
  nh.param<int>("publish/publish_img_stride", publish_img_stride_, 1);

  nh.param<int>("lio/map_update_stride", lio_map_update_stride_, 1);
  nh.param<bool>("lio/force_voxel_map_update", lio_force_voxel_map_update_, true);
  nh.param<double>("lio/force_map_update_interval", lio_force_map_update_interval_, 0.3);
  nh.param<int>("lio/force_map_update_lidar_frames", lio_force_map_update_lidar_frames_, 3);

  nh.param<bool>("debug/print_console_timing_en", print_console_timing_en_, true);
  nh.param<int>("debug/print_console_timing_stride", print_console_timing_stride_, 1);

  nh.param<bool>("aruco_landmarks/aruco_landmarks_en", aruco_landmarks_en, false);

  nh.param<bool>("adaptive_selector/en", adaptive_visual_selector_en, false);
  keyframe_trans_thresh_min_ = 0.08;
  keyframe_trans_thresh_max_ = 0.18;
  keyframe_rot_thresh_min_deg_ = 1.2;
  keyframe_rot_thresh_max_deg_ = 2.5;
  keyframe_constraint_ratio_full_ = 0.2;
  keyframe_max_skip_frames_ = 4;
  nh.param<double>("adaptive_selector/trans_thresh_min", keyframe_trans_thresh_min_, keyframe_trans_thresh_min_);
  nh.param<double>("adaptive_selector/trans_thresh_max", keyframe_trans_thresh_max_, keyframe_trans_thresh_max_);
  nh.param<double>("adaptive_selector/rot_thresh_min_deg", keyframe_rot_thresh_min_deg_, keyframe_rot_thresh_min_deg_);
  nh.param<double>("adaptive_selector/rot_thresh_max_deg", keyframe_rot_thresh_max_deg_, keyframe_rot_thresh_max_deg_);
  nh.param<double>("adaptive_selector/constraint_ratio_full", keyframe_constraint_ratio_full_, keyframe_constraint_ratio_full_);
  nh.param<int>("adaptive_selector/max_skip_frames", keyframe_max_skip_frames_, keyframe_max_skip_frames_);
  keyframe_trans_thresh_min_ = std::max(0.0, keyframe_trans_thresh_min_);
  keyframe_trans_thresh_max_ = std::max(keyframe_trans_thresh_min_, keyframe_trans_thresh_max_);
  keyframe_rot_thresh_min_deg_ = std::max(0.0, keyframe_rot_thresh_min_deg_);
  keyframe_rot_thresh_max_deg_ = std::max(keyframe_rot_thresh_min_deg_, keyframe_rot_thresh_max_deg_);
  keyframe_constraint_ratio_full_ = std::max(1e-6, keyframe_constraint_ratio_full_);
  keyframe_max_skip_frames_ = std::max(0, keyframe_max_skip_frames_);

  keyframe_trans_thresh_min_nominal_ = keyframe_trans_thresh_min_;
  keyframe_trans_thresh_max_nominal_ = keyframe_trans_thresh_max_;
  keyframe_rot_thresh_min_deg_nominal_ = keyframe_rot_thresh_min_deg_;
  keyframe_rot_thresh_max_deg_nominal_ = keyframe_rot_thresh_max_deg_;
  keyframe_max_skip_frames_nominal_ = keyframe_max_skip_frames_;
  vio_max_iterations_nominal_ = max_iterations;

  pub_scan_num = std::max(1, pub_scan_num);
  publish_img_stride_ = std::max(1, publish_img_stride_);
  lio_map_update_stride_ = std::max(1, lio_map_update_stride_);
  lio_force_map_update_interval_ = std::max(0.0, lio_force_map_update_interval_);
  lio_force_map_update_lidar_frames_ = std::max(1, lio_force_map_update_lidar_frames_);
  print_console_timing_stride_ = std::max(1, print_console_timing_stride_);

  pub_scan_num_nominal_ = pub_scan_num;
  dense_map_en_nominal_ = dense_map_en;
  pcd_save_en_nominal_ = pcd_save_en;
  colorize_cloud_en_nominal_ = colorize_cloud_en_;

  nh.param<bool>("runtime_guard/enable", runtime_guard_en_, true);
  nh.param<double>("runtime_guard/frame_time_budget_s", frame_time_budget_s_, frame_time_budget_s_);
  if (!experimental_features_enable_)
  {
    // ponytail: baseline mode must not let experiment guards change FAST-LIVO2 state, maps, or publishing.
    deg_guard_enable_ = false;
    deg_guard_enable_z_soft_constraint_ = false;
    deg_guard_enable_nhc_ = false;
    deg_guard_enable_backward_guard_ = false;
    deg_guard_enable_corridor_detection_ = false;
    deg_guard_enable_adaptive_sensor_weighting_ = false;
    deg_guard_reject_large_update_in_degenerate_ = false;
    safety_guard_enable_ = false;
    safety_fail_safe_mode_ = false;
    local_reinit_enable_ = false;
    debug_fixed_degraded_intervals_enable_ = false;
    degraded_bootstrap_enable_ = false;
    disable_visual_map_in_degraded_hold_ = false;
    disable_voxel_map_in_degraded_hold_ = false;
    fixed_degraded_trigger_mode_ = "disabled";
    local_mode_ = "NORMAL";
    local_reinit_reason_ = "experimental_disabled";
    corridor_prior_enable_ = false;
    corridor_prior_action_ = "none";
    corridor_prior_update_voxel_map_enabled_ = true;
    corridor_prior_visual_map_update_enabled_ = true;
    runtime_guard_en_ = false;
    adaptive_visual_selector_en = false;
    lio_freeze_state_when_degenerate_ = false;
    lio_freeze_state_ready_ = false;
    lio_state_jump_guard_en_ = false;
    uwb_output_pos_offset_.setZero();
    uwb_output_target_offset_.setZero();
    vio_visual_update_guard_en_ = false;
    vio_image_quality_gate_en_ = false;
    vio_visual_patch_quality_gate_en_ = false;
    visual_map_prune_en = false;
    visual_map_max_voxels = 0;
    visual_map_max_points_per_voxel = 0;
    visual_map_max_total_points = 0;
    visual_map_max_add_per_frame_ = 1000000000;
    visual_map_min_shi_tomasi_score_ = 0.0;
    current_decision_ = UpdateDecision();
    update_decision_ready_ = false;
  }
  if (frame_time_budget_s_ <= 0.0)
  {
    ROS_WARN("[RuntimeGuard] Invalid frame_time_budget_s=%.6f, fallback to 0.100000 s", frame_time_budget_s_);
    frame_time_budget_s_ = 0.1;
  }
  ROS_INFO("[Experimental] enable=%d", static_cast<int>(experimental_features_enable_));
  ROS_INFO("[RuntimeGuard] enable=%d, frame_time_budget_s=%.6f s", static_cast<int>(runtime_guard_en_), frame_time_budget_s_);
  ROS_INFO("[DEGEN_GUARD] enable=%d adaptive_weight=%d z_soft=%d nhc=%d backward=%d corridor=%d action=%s log_file=%s",
           static_cast<int>(deg_guard_enable_),
           static_cast<int>(deg_guard_enable_adaptive_sensor_weighting_),
           static_cast<int>(deg_guard_enable_z_soft_constraint_),
           static_cast<int>(deg_guard_enable_nhc_),
           static_cast<int>(deg_guard_enable_backward_guard_),
           static_cast<int>(deg_guard_enable_corridor_detection_),
           deg_guard_backward_action_.c_str(),
           deg_guard_log_file_.c_str());
  ROS_INFO("[AdaptiveSelector] enable=%d, trans=[%.3f %.3f] m, rot=[%.3f %.3f] deg, ratio_full=%.3f, max_skip=%d",
           static_cast<int>(adaptive_visual_selector_en),
           keyframe_trans_thresh_min_,
           keyframe_trans_thresh_max_,
           keyframe_rot_thresh_min_deg_,
           keyframe_rot_thresh_max_deg_,
           keyframe_constraint_ratio_full_,
           keyframe_max_skip_frames_);

  pub_scan_num_degraded_ = std::max(1, pub_scan_num_degraded_);
  runtime_over_budget_trigger_frames_ = std::max(1, runtime_over_budget_trigger_frames_);
  runtime_recover_trigger_frames_ = std::max(1, runtime_recover_trigger_frames_);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::updateRuntimeGuard(double frame_time_s)
{
  if (!experimental_features_enable_) return;
  if (!runtime_guard_en_) return;
  if (deterministic_mode_)
  {
    ROS_WARN_THROTTLE(2.0,
                      "[RuntimeGuard] deterministic_mode=true: frame_time=%.4f s budget=%.4f s, log-only no algorithm branch change.",
                      frame_time_s,
                      frame_time_budget_s_);
    return;
  }

  if (frame_time_s > frame_time_budget_s_)
  {
    runtime_over_budget_count_++;
    runtime_under_budget_count_ = 0;
  }
  else
  {
    runtime_under_budget_count_++;
    runtime_over_budget_count_ = 0;
  }

  if (!runtime_degraded_mode_ && runtime_over_budget_count_ >= std::max(1, runtime_over_budget_trigger_frames_))
  {
    runtime_degraded_mode_ = true;
    runtime_over_budget_count_ = 0;
    runtime_under_budget_count_ = 0;

    pub_scan_num = std::max(pub_scan_num_nominal_, pub_scan_num_degraded_);
    if (disable_dense_map_in_degraded_) dense_map_en = false;
    if (disable_pcd_save_in_degraded_) pcd_save_en = false;
    if (disable_colorize_cloud_in_degraded_) colorize_cloud_en_ = false;
    suppress_image_pub_ = disable_image_publish_in_degraded_;
    publish_img_counter_ = 0;

    if (vio_manager)
    {
      const int degraded_iters = std::max(1, std::min(vio_max_iterations_nominal_, vio_max_iterations_degraded_));
      vio_manager->max_iterations = degraded_iters;
    }

    if (adaptive_visual_selector_en)
    {
      keyframe_trans_thresh_min_ = keyframe_trans_thresh_min_nominal_ * std::max(1.0, keyframe_trans_scale_degraded_);
      keyframe_trans_thresh_max_ = keyframe_trans_thresh_max_nominal_ * std::max(1.0, keyframe_trans_scale_degraded_);
      keyframe_rot_thresh_min_deg_ = keyframe_rot_thresh_min_deg_nominal_ * std::max(1.0, keyframe_rot_scale_degraded_);
      keyframe_rot_thresh_max_deg_ = keyframe_rot_thresh_max_deg_nominal_ * std::max(1.0, keyframe_rot_scale_degraded_);
      keyframe_max_skip_frames_ = std::max(keyframe_max_skip_frames_nominal_, keyframe_max_skip_frames_degraded_);
      skipped_visual_frames_ = 0;
    }

    ROS_WARN("[RuntimeGuard] Enter degraded mode, frame_time=%.4f s > budget=%.4f s, pub_scan_num=%d, dense_map=%d, pcd_save=%d, colorize=%d, img_pub=%d",
             frame_time_s,
             frame_time_budget_s_,
             pub_scan_num,
             static_cast<int>(dense_map_en),
             static_cast<int>(pcd_save_en),
             static_cast<int>(colorize_cloud_en_),
             static_cast<int>(!suppress_image_pub_));
    return;
  }

  if (runtime_degraded_mode_ && runtime_under_budget_count_ >= std::max(1, runtime_recover_trigger_frames_))
  {
    runtime_degraded_mode_ = false;
    runtime_over_budget_count_ = 0;
    runtime_under_budget_count_ = 0;

    pub_scan_num = std::max(1, pub_scan_num_nominal_);
    dense_map_en = dense_map_en_nominal_;
    pcd_save_en = pcd_save_en_nominal_;
    colorize_cloud_en_ = colorize_cloud_en_nominal_;
    suppress_image_pub_ = false;
    publish_img_counter_ = 0;

    if (vio_manager)
    {
      vio_manager->max_iterations = std::max(1, vio_max_iterations_nominal_);
    }

    keyframe_trans_thresh_min_ = keyframe_trans_thresh_min_nominal_;
    keyframe_trans_thresh_max_ = keyframe_trans_thresh_max_nominal_;
    keyframe_rot_thresh_min_deg_ = keyframe_rot_thresh_min_deg_nominal_;
    keyframe_rot_thresh_max_deg_ = keyframe_rot_thresh_max_deg_nominal_;
    keyframe_max_skip_frames_ = keyframe_max_skip_frames_nominal_;
    skipped_visual_frames_ = 0;

    ROS_INFO("[RuntimeGuard] Recover nominal mode, frame_time=%.4f s, pub_scan_num=%d, dense_map=%d, pcd_save=%d, colorize=%d, img_pub=%d",
             frame_time_s,
             pub_scan_num,
             static_cast<int>(dense_map_en),
             static_cast<int>(pcd_save_en),
             static_cast<int>(colorize_cloud_en_),
             static_cast<int>(!suppress_image_pub_));
  }
}

void LIVMapper::initializeComponents(ros::NodeHandle &nh) 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);
  voxelmap_manager->adaptive_sensor_weighting_en_ = deg_guard_enable_ && deg_guard_enable_adaptive_sensor_weighting_;
  voxelmap_manager->adaptive_low_feature_noise_scale_ = deg_guard_adaptive_lio_low_feature_noise_scale_;
  voxelmap_manager->adaptive_high_residual_noise_scale_ = deg_guard_adaptive_lio_high_residual_noise_scale_;
  voxelmap_manager->adaptive_residual_ref_ = deg_guard_adaptive_lio_residual_ref_;
  voxelmap_manager->adaptive_max_noise_scale_ = deg_guard_adaptive_max_noise_scale_;
  voxelmap_manager->adaptive_min_lidar_features_ = deg_guard_min_lidar_features_;

  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam)) throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->adaptive_sensor_weighting_en = deg_guard_enable_ && deg_guard_enable_adaptive_sensor_weighting_;
  vio_manager->adaptive_low_track_noise_scale = deg_guard_adaptive_vio_low_track_noise_scale_;
  vio_manager->adaptive_max_noise_scale = std::max(deg_guard_adaptive_max_noise_scale_,
                                                   deg_guard_adaptive_vio_low_track_noise_scale_);
  vio_manager->adaptive_min_tracked_points = std::max(deg_guard_min_visual_tracked_points_,
                                                      deg_guard_vio_low_feature_tracked_points_ + 1);
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->min_retrieve_points = deg_guard_enable_ ? deg_guard_vio_skip_min_tracked_points_ : vio_min_retrieve_points_;
  vio_manager->min_update_meas = deg_guard_enable_ ? deg_guard_vio_min_update_meas_ : vio_min_update_meas_;
  vio_manager->low_track_force_update_stride = vio_low_track_force_update_stride_;
  vio_manager->low_track_force_min_points = vio_low_track_force_min_points_;
  vio_manager->deterministic_visual_update_en = vio_deterministic_visual_update_en_;
  vio_manager->deterministic_pixel_snap_en = deterministic_pixel_snap_en_;
  vio_manager->deterministic_camera_point_snap_en = deterministic_camera_point_snap_en_;
  vio_manager->deterministic_contiguous_image_copy_en = deterministic_contiguous_image_copy_en_;
  vio_manager->deterministic_visual_voxel_key_sort_en = deterministic_visual_voxel_key_sort_en_;
  vio_manager->visual_update_guard_en = vio_visual_update_guard_en_;
  vio_manager->visual_update_max_trans_m = vio_visual_update_max_trans_m_;
  vio_manager->visual_update_max_rot_deg = vio_visual_update_max_rot_deg_;
  vio_manager->visual_update_max_trans_rate_mps = vio_visual_update_max_trans_rate_mps_;
  vio_manager->visual_update_max_rot_rate_degps = vio_visual_update_max_rot_rate_degps_;
  vio_manager->visual_update_max_backward_rate_mps = vio_visual_update_max_backward_rate_mps_;
  vio_manager->visual_update_max_lateral_rate_mps = vio_visual_update_max_lateral_rate_mps_;
  vio_manager->visual_update_max_backward_m = vio_visual_update_max_backward_m_;
  vio_manager->visual_update_max_backward_ratio = vio_visual_update_max_backward_ratio_;
  vio_manager->visual_update_backward_abs_floor_m = vio_visual_update_backward_abs_floor_m_;
  vio_manager->visual_update_max_lateral_m = vio_visual_update_max_lateral_m_;
  vio_manager->visual_update_max_lateral_ratio = vio_visual_update_max_lateral_ratio_;
  vio_manager->visual_update_max_exposure_delta = vio_visual_update_max_exposure_delta_;
  vio_manager->visual_update_large_update_guard_action = deg_guard_enable_ ? vio_visual_update_large_update_guard_action_ : "reject_update";
  vio_manager->visual_update_large_rotation_action = deg_guard_enable_ ? vio_visual_update_large_rotation_action_ : "reject_update";
  vio_manager->reject_visual_large_rotation = deg_guard_enable_ && (vio_reject_visual_large_rotation_ || deg_guard_use_vio_large_rotation_for_reject_);
  vio_manager->use_vio_large_rotation_for_reject = deg_guard_enable_ && deg_guard_use_vio_large_rotation_for_reject_;
  vio_manager->visual_update_large_rotation_noise_scale = vio_visual_update_large_rotation_noise_scale_;
  vio_manager->visual_update_backward_guard_action = deg_guard_enable_ ? vio_visual_update_backward_guard_action_ : "reject_update";
  vio_manager->visual_update_lateral_guard_action = deg_guard_enable_ ? vio_visual_update_lateral_guard_action_ : "reject_update";
  vio_manager->visual_update_exposure_guard_action = deg_guard_enable_ ? vio_visual_update_exposure_guard_action_ : "reject_update";
  vio_manager->visual_update_nonfinite_guard_action = deg_guard_enable_ ? vio_visual_update_nonfinite_guard_action_ : "reject_update";
  vio_manager->image_quality_gate_en = vio_image_quality_gate_en_;
  vio_manager->image_quality_max_saturated_fraction = vio_image_quality_max_saturated_fraction_;
  vio_manager->image_quality_max_tile_saturated_fraction = vio_image_quality_max_tile_saturated_fraction_;
  vio_manager->image_quality_max_dark_fraction = vio_image_quality_max_dark_fraction_;
  vio_manager->image_quality_min_intensity_std = vio_image_quality_min_intensity_std_;
  vio_manager->visual_patch_quality_gate_en = vio_visual_patch_quality_gate_en_;
  vio_manager->visual_patch_max_saturated_fraction = vio_visual_patch_max_saturated_fraction_;
  vio_manager->visual_patch_min_intensity_std = vio_visual_patch_min_intensity_std_;
  vio_manager->image_quality_saturated_pixel_value = vio_image_quality_saturated_pixel_value_;
  vio_manager->image_quality_dark_pixel_value = vio_image_quality_dark_pixel_value_;
  vio_manager->image_quality_tile_rows = vio_image_quality_tile_rows_;
  vio_manager->image_quality_tile_cols = vio_image_quality_tile_cols_;
  vio_manager->max_state_update_rot_deg = vio_max_state_update_rot_deg_;
  vio_manager->max_state_update_trans_m = vio_max_state_update_trans_m_;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->visual_map_prune_en = visual_map_prune_en;
  vio_manager->visual_map_max_voxels = visual_map_max_voxels;
  vio_manager->visual_map_max_points_per_voxel = visual_map_max_points_per_voxel;
  vio_manager->visual_map_max_total_points = visual_map_max_total_points;
  vio_manager->visual_map_max_add_per_frame = visual_map_max_add_per_frame_;
  vio_manager->visual_map_min_shi_tomasi_score = static_cast<float>(visual_map_min_shi_tomasi_score_);
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->aruco_landmarks_en = aruco_landmarks_en;
  vio_manager->timing_log_dir = save_path;
  vio_manager->timing_log_enable = save_log_en;
  vio_manager->initializeVIO(nh);
  initializeUdpReporter();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);
  p_imu->set_log_dir(save_path);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(save_path + "points3D.txt", std::ios::out);
  if(save_log_en) fout_pcd_pos.open(save_path + "scans_pos.json", std::ios::out);
	  if (diagnostics_level_ == "verbose" &&
	      (deg_guard_enable_ || safety_guard_enable_ || corridor_prior_enable_) &&
	      !deg_guard_log_file_.empty())
  {
    degeneracy_guard_log_.open(deg_guard_log_file_, std::ios::out);
    if (degeneracy_guard_log_.is_open())
    {
		      degeneracy_guard_log_ << "timestamp,frame_id,stage,sensor_type,mode,local_mode,deterministic_mode,dt,"
		                            << "local_elapsed,bag_start_offset,bag_elapsed,in_fixed_degraded_window,fixed_degraded_reason,"
		                            << "det_lio_update,det_vio_update,det_uwb_update,uwb_anchor_ids,"
	                            << "uwb_xy_correction,uwb_z_before_clamp,uwb_z_after_clamp,"
	                            << "local_map_cleared,visual_map_cleared,tracker_reset,"
	                            << "lio_bootstrap_frames,vio_bootstrap_frames,vio_unavailable,lio_weak,"
	                            << "position_x,position_y,position_z,"
	                            << "velocity_world_x,velocity_world_y,velocity_world_z,"
	                            << "velocity_body_x,velocity_body_y,velocity_body_z,"
	                            << "speed,quaternion_norm,frame_translation,frame_rotation_deg,backward_distance_in_window,"
	                            << "corridor_axis_x,corridor_axis_y,corridor_axis_z,"
	                            << "corridor_progress,corridor_progress_delta_1s,corridor_progress_delta_window,"
	                            << "corridor_backward_distance_window,corridor_backward_distance_fail_window,"
	                            << "corridor_action,corridor_update_voxel_map_enabled,corridor_visual_map_update_enabled,"
	                            << "z,z_residual,forward_progress,is_backward_slip,"
                            << "lio_feature_count,visual_tracked_points,"
                            << "update_translation_norm,update_translation_rate_mps,"
                            << "update_yaw_deg,update_yaw_rate_degps,"
                            << "visual_rotation_update_deg,visual_rotation_rate_degps,"
                            << "z_correction_norm,nhc_correction_norm,"
                            << "lio_noise_scale,vio_noise_scale,weight_reason,"
                            << "lio_update_executed,lio_downweighted,lio_update_voxel_map,lio_voxel_map_skip_reason,"
                            << "vio_skip_affects_global_degenerate,"
                            << "update_status,reject_reason,action,reason,final_pose_delta\n";
      ROS_INFO("[DEGEN_GUARD] log_file=%s", deg_guard_log_file_.c_str());
    }
    else
    {
      ROS_WARN("[DEGEN_GUARD] Failed to open log_file=%s", deg_guard_log_file_.c_str());
    }
  }
}

void LIVMapper::initializeUdpReporter()
{
  if (!udp_report_en) return;
  if (udp_target_ip_.empty() || udp_report_port_ <= 0 || udp_report_port_ > 65535)
  {
    ROS_WARN("[UDP] Invalid target config, reporting disabled.");
    udp_report_en = false;
    return;
  }

  udp_socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_socket_fd_ < 0)
  {
    ROS_WARN("[UDP] Failed to create socket, reporting disabled.");
    udp_report_en = false;
    return;
  }

  udp_target_addr_ = {};
  udp_target_addr_.sin_family = AF_INET;
  udp_target_addr_.sin_port = htons(static_cast<uint16_t>(udp_report_port_));
  if (::inet_pton(AF_INET, udp_target_ip_.c_str(), &udp_target_addr_.sin_addr) != 1)
  {
    ROS_WARN("[UDP] Invalid target IP '%s', reporting disabled.", udp_target_ip_.c_str());
    ::close(udp_socket_fd_);
    udp_socket_fd_ = -1;
    udp_report_en = false;
    return;
  }

  udp_socket_ready_ = true;
  sendUdpMessage(std::string("DEVICE_ID ") + udp_device_id_);
  ROS_INFO("[UDP] Reporting enabled for %s:%d", udp_target_ip_.c_str(), udp_report_port_);
}

void LIVMapper::sendUdpMessage(const std::string &message)
{
  if (!udp_report_en || !udp_socket_ready_ || udp_socket_fd_ < 0 || message.empty()) return;

  const ssize_t sent = ::sendto(udp_socket_fd_, message.c_str(), message.size(), 0,
                                reinterpret_cast<const sockaddr *>(&udp_target_addr_), sizeof(udp_target_addr_));
  if (sent < 0)
  {
    ROS_WARN_THROTTLE(5.0, "[UDP] Failed to send packet to %s:%d.", udp_target_ip_.c_str(), udp_report_port_);
  }
}

void LIVMapper::sendUdpPose(const Eigen::Vector3d &position)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6)
      << "POSE " << ros::Time::now().toSec() << ' '
      << position.x() << ' ' << position.y() << ' ' << position.z();
  sendUdpMessage(oss.str());
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, sub_lidar_queue_size_, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, sub_lidar_queue_size_, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, sub_imu_queue_size_, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, sub_img_queue_size_, &LIVMapper::img_cbk, this);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/mapping/globalMap", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/mapping/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment()
{
  if (!p_imu->imu_need_init && !gravity_align_finished)
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::blendDegeneracyGuardUpdate(const StatesGroup &state_before_update,
                                           const StatesGroup &state_after_update,
                                           double scale)
{
  scale = std::max(0.0, std::min(1.0, scale));
  StatesGroup state_after_copy = state_after_update;
  VD(DIM_STATE) dx = state_after_copy - state_before_update;
  dx *= scale;
  _state = state_before_update;
  _state += dx;
  _state.cov = state_before_update.cov + scale * (state_after_update.cov - state_before_update.cov);
  _state.cov = 0.5 * (_state.cov + _state.cov.transpose());
}

void LIVMapper::resetCorridorMotionPrior()
{
  corridor_prior_started_ = false;
  corridor_prior_axis_ready_ = false;
  corridor_prior_axis_failed_ = false;
  corridor_prior_entry_time_ = -1.0;
  corridor_prior_entry_pos_ = V3D::Zero();
  corridor_prior_axis_ = V3D::UnitX();
  corridor_prior_progress_buffer_.clear();
  corridor_prior_progress_ = 0.0;
  corridor_prior_progress_delta_1s_ = 0.0;
  corridor_prior_progress_delta_window_ = 0.0;
  corridor_prior_backward_distance_window_ = 0.0;
  corridor_prior_backward_distance_fail_window_ = 0.0;
  corridor_prior_action_ = "none";
  corridor_prior_update_voxel_map_enabled_ = true;
  corridor_prior_visual_map_update_enabled_ = true;
}

void LIVMapper::updateCorridorMotionPrior(const char *stage, const StatesGroup &state)
{
  corridor_prior_action_ = "none";
  corridor_prior_update_voxel_map_enabled_ = true;
  corridor_prior_visual_map_update_enabled_ = true;
  corridor_prior_progress_delta_1s_ = 0.0;
  corridor_prior_progress_delta_window_ = 0.0;
  corridor_prior_backward_distance_window_ = 0.0;
  corridor_prior_backward_distance_fail_window_ = 0.0;

  if (!experimental_features_enable_ || !corridor_prior_enable_ || isDegradedHoldMode() || isBootstrapMode() ||
      (p_imu && p_imu->imu_need_init) ||
      (gravity_align_en && !gravity_align_finished) || !state.pos_end.allFinite())
  {
    return;
  }

  if (corridor_prior_only_in_degenerate_ && !deg_guard_corridor_degenerate_ && !corridor_prior_started_)
  {
    return;
  }

  const double now = LidarMeasures.last_lio_update_time;
  if (!std::isfinite(now)) return;

  if (!corridor_prior_started_)
  {
    corridor_prior_started_ = true;
    corridor_prior_axis_ready_ = false;
    corridor_prior_axis_failed_ = false;
    corridor_prior_entry_time_ = now;
    corridor_prior_entry_pos_ = state.pos_end;
    corridor_prior_axis_ = V3D::UnitX();
    corridor_prior_progress_buffer_.clear();
    ROS_WARN("[CORRIDOR_PRIOR] start axis estimation stage=%s t=%.6f pos=[%.3f %.3f %.3f]",
             stage ? stage : "state",
             now,
             state.pos_end[0], state.pos_end[1], state.pos_end[2]);
  }

  if (!corridor_prior_axis_ready_ && !corridor_prior_axis_failed_)
  {
    const V3D net_motion = state.pos_end - corridor_prior_entry_pos_;
    const double net_motion_norm = net_motion.norm();
    const double elapsed = now - corridor_prior_entry_time_;
    if (elapsed >= corridor_prior_axis_estimation_sec_ &&
        net_motion_norm >= corridor_prior_min_axis_motion_)
    {
      corridor_prior_axis_ = net_motion / std::max(1e-9, net_motion_norm);
      corridor_prior_axis_ready_ = true;
      corridor_prior_progress_buffer_.clear();
      ROS_WARN("[CORRIDOR_PRIOR] corridor_axis ready axis=[%.6f %.6f %.6f] elapsed=%.3f net_motion=%.3f",
               corridor_prior_axis_[0], corridor_prior_axis_[1], corridor_prior_axis_[2],
               elapsed,
               net_motion_norm);
    }
    else if (elapsed >= corridor_prior_axis_estimation_max_sec_)
    {
      corridor_prior_axis_failed_ = true;
      ROS_WARN("[CORRIDOR_PRIOR] corridor_axis disabled: net_motion=%.3f < min_axis_motion=%.3f elapsed=%.3f",
               net_motion_norm,
               corridor_prior_min_axis_motion_,
               elapsed);
    }
  }

  if (!corridor_prior_axis_ready_) return;

  corridor_prior_progress_ = (state.pos_end - corridor_prior_entry_pos_).dot(corridor_prior_axis_);
  corridor_prior_progress_buffer_.push_back({now, corridor_prior_progress_});
  const double max_keep_sec = std::max(corridor_prior_fail_safe_window_sec_,
                                       std::max(corridor_prior_backward_window_sec_, 1.0));
  while (!corridor_prior_progress_buffer_.empty() &&
         now - corridor_prior_progress_buffer_.front().first > max_keep_sec)
  {
    corridor_prior_progress_buffer_.pop_front();
  }

  auto computeWindow = [&](double window_sec,
                           double *delta_from_oldest,
                           double *backward_from_max) {
    bool has_sample = false;
    double oldest_progress = corridor_prior_progress_;
    double max_progress = corridor_prior_progress_;
    for (const auto &sample : corridor_prior_progress_buffer_)
    {
      if (now - sample.first <= window_sec)
      {
        if (!has_sample)
        {
          oldest_progress = sample.second;
          max_progress = sample.second;
          has_sample = true;
        }
        max_progress = std::max(max_progress, sample.second);
      }
    }
    if (!has_sample)
    {
      oldest_progress = corridor_prior_progress_;
      max_progress = corridor_prior_progress_;
    }
    if (delta_from_oldest) *delta_from_oldest = corridor_prior_progress_ - oldest_progress;
    if (backward_from_max) *backward_from_max = std::max(0.0, max_progress - corridor_prior_progress_);
  };

  computeWindow(1.0, &corridor_prior_progress_delta_1s_, nullptr);
  computeWindow(corridor_prior_backward_window_sec_,
                &corridor_prior_progress_delta_window_,
                &corridor_prior_backward_distance_window_);
  computeWindow(corridor_prior_fail_safe_window_sec_,
                nullptr,
                &corridor_prior_backward_distance_fail_window_);

  if (corridor_prior_backward_action_ == "log_only")
  {
    corridor_prior_action_ = (corridor_prior_backward_distance_window_ > corridor_prior_backward_distance_threshold_) ?
                             "log_only" : "none";
  }
  else if (corridor_prior_backward_action_ == "downweight")
  {
    if (corridor_prior_backward_distance_fail_window_ > corridor_prior_fail_safe_backward_distance_threshold_)
    {
      corridor_prior_action_ = "local_reinit";
    }
    else if (corridor_prior_backward_distance_window_ > corridor_prior_backward_distance_threshold_)
    {
      corridor_prior_action_ = "block_map_update";
    }
  }
  else if (corridor_prior_backward_action_ == "fail_safe")
  {
    if (corridor_prior_backward_distance_window_ > corridor_prior_backward_distance_threshold_ ||
        corridor_prior_backward_distance_fail_window_ > corridor_prior_fail_safe_backward_distance_threshold_)
    {
      corridor_prior_action_ = "local_reinit";
    }
  }

  if (corridor_prior_action_ == "downweight" || corridor_prior_action_ == "block_map_update")
  {
    corridor_prior_update_voxel_map_enabled_ = !corridor_prior_disable_map_update_on_downweight_;
    corridor_prior_visual_map_update_enabled_ = !corridor_prior_disable_visual_map_update_on_downweight_;
    ROS_WARN_THROTTLE(0.5,
	                      "[CORRIDOR_PRIOR] BACKWARD_DOWNWEIGHT stage=%s action=%s progress=%.3f d1s=%.3f dwin=%.3f back_win=%.3f/%.3f axis=[%.3f %.3f %.3f]",
	                      stage ? stage : "state",
	                      corridor_prior_action_.c_str(),
	                      corridor_prior_progress_,
                      corridor_prior_progress_delta_1s_,
                      corridor_prior_progress_delta_window_,
                      corridor_prior_backward_distance_window_,
                      corridor_prior_backward_distance_threshold_,
                      corridor_prior_axis_[0], corridor_prior_axis_[1], corridor_prior_axis_[2]);
  }
  else if (corridor_prior_action_ == "local_reinit")
  {
    const double progress_keep = corridor_prior_progress_;
    const double progress_delta_1s_keep = corridor_prior_progress_delta_1s_;
    const double progress_delta_window_keep = corridor_prior_progress_delta_window_;
    const double backward_window_keep = corridor_prior_backward_distance_window_;
    const double backward_fail_keep = corridor_prior_backward_distance_fail_window_;
    const V3D axis_keep = corridor_prior_axis_;
    corridor_prior_update_voxel_map_enabled_ = false;
    corridor_prior_visual_map_update_enabled_ = false;
    ROS_ERROR("[CORRIDOR_PRIOR] BACKWARD_LOCAL_REINIT stage=%s progress=%.3f back_fail=%.3f/%.3f axis=[%.3f %.3f %.3f]",
              stage ? stage : "state",
              corridor_prior_progress_,
              corridor_prior_backward_distance_fail_window_,
              corridor_prior_fail_safe_backward_distance_threshold_,
              corridor_prior_axis_[0], corridor_prior_axis_[1], corridor_prior_axis_[2]);
    if (local_reinit_enable_) enterLocalReinitMode("LOCAL_REINIT", "BACKWARD_LOCAL_REINIT");
    else enterFailSafe("BACKWARD_LOCAL_REINIT", &state);
    corridor_prior_action_ = "local_reinit";
    corridor_prior_update_voxel_map_enabled_ = false;
    corridor_prior_visual_map_update_enabled_ = false;
    corridor_prior_progress_ = progress_keep;
    corridor_prior_progress_delta_1s_ = progress_delta_1s_keep;
    corridor_prior_progress_delta_window_ = progress_delta_window_keep;
    corridor_prior_backward_distance_window_ = backward_window_keep;
    corridor_prior_backward_distance_fail_window_ = backward_fail_keep;
    corridor_prior_axis_ = axis_keep;
  }
  else if (corridor_prior_action_ == "log_only")
  {
    ROS_WARN_THROTTLE(0.5,
                      "[CORRIDOR_PRIOR] BACKWARD_LOG_ONLY stage=%s progress=%.3f back_win=%.3f/%.3f",
                      stage ? stage : "state",
                      corridor_prior_progress_,
                      corridor_prior_backward_distance_window_,
                      corridor_prior_backward_distance_threshold_);
  }
}

double LIVMapper::updateAdaptiveSensorNoiseScale(const char *sensor)
{
  if (!experimental_features_enable_) return 1.0;
  const std::string sensor_name = sensor ? sensor : "state";
  const bool is_lio = sensor_name.find("LIO") != std::string::npos || sensor_name.find("LO") != std::string::npos;
  const bool is_vio = sensor_name.find("VIO") != std::string::npos;

  double scale = 1.0;
  std::ostringstream reason;
  bool has_reason = false;
  auto add_reason = [&](const std::string &item) {
    if (has_reason) reason << ";";
    reason << item;
    has_reason = true;
  };

  if (deg_guard_enable_ && deg_guard_enable_adaptive_sensor_weighting_)
  {
    scale = is_lio ? deg_guard_adaptive_lio_base_noise_scale_ :
            is_vio ? deg_guard_adaptive_vio_base_noise_scale_ : 1.0;
    if (scale > 1.0) add_reason("base");

    if (deg_guard_corridor_degenerate_)
    {
      scale *= is_lio ? deg_guard_degenerate_lio_noise_scale_ :
               is_vio ? deg_guard_degenerate_vio_noise_scale_ : 1.0;
      add_reason("corridor_degenerate");
    }
	    if (deg_guard_last_backward_slip_)
	    {
	      scale *= is_lio ? deg_guard_degenerate_lio_noise_scale_ :
	               is_vio ? deg_guard_degenerate_vio_noise_scale_ : 1.0;
	      add_reason("backward_slip");
	    }
	    if (!has_reason) add_reason("nominal");
	  }
	  else
	  {
	    add_reason("off");
	  }
		  if (corridor_prior_enable_ &&
		      (corridor_prior_action_ == "downweight" || corridor_prior_action_ == "block_map_update"))
	  {
	    scale *= is_lio ? corridor_prior_lio_downweight_scale_ :
	             is_vio ? corridor_prior_vio_downweight_scale_ : 1.0;
	    add_reason("corridor_backward_prior");
	  }
	  scale = std::min(std::max(1.0, scale), deg_guard_adaptive_max_noise_scale_);

  deg_guard_last_weight_reason_ = reason.str();
  if (is_lio) deg_guard_last_lio_noise_scale_ = scale;
  if (is_vio) deg_guard_last_vio_noise_scale_ = scale;

  if (deg_guard_enable_)
  {
    ROS_INFO_THROTTLE(1.0,
                      "[DEGEN_GUARD][AdaptiveWeight] sensor=%s external_noise_scale=%.3f reason=%s degenerate=%d backward=%d",
                      sensor_name.c_str(),
                      scale,
                      deg_guard_last_weight_reason_.c_str(),
                      static_cast<int>(deg_guard_corridor_degenerate_),
                      static_cast<int>(deg_guard_last_backward_slip_));
  }
  return scale;
}

bool LIVMapper::applyScalarPseudoMeasurement(StatesGroup &state,
                                             const Eigen::Matrix<double, 1, DIM_STATE> &H,
                                             double residual,
                                             double sigma,
                                             double gain,
                                             double *correction_norm)
{
  if (correction_norm) *correction_norm = 0.0;
  if (!std::isfinite(residual) || !std::isfinite(sigma) || sigma <= 0.0 || gain <= 0.0) return false;
  if (!state.cov.allFinite()) return false;

  const double R = sigma * sigma;
  const double H_norm2 = H.squaredNorm();
  if (H_norm2 < 1e-12) return false;

  const MD(DIM_STATE, DIM_STATE) P_before = state.cov;
  const double S = (H * P_before * H.transpose())(0, 0) + R;
  VD(DIM_STATE) dx = VD(DIM_STATE)::Zero();
  if (std::isfinite(S) && S > 1e-12)
  {
    const VD(DIM_STATE) K = P_before * H.transpose() / S;
    const VD(DIM_STATE) K_eff = gain * K;
    dx = -K_eff * residual;
    if (!dx.allFinite()) return false;
    state += dx;

    const MD(DIM_STATE, DIM_STATE) I_STATE = MD(DIM_STATE, DIM_STATE)::Identity();
    const MD(DIM_STATE, DIM_STATE) I_KH = I_STATE - K_eff * H;
    state.cov = I_KH * P_before * I_KH.transpose() + (K_eff * K_eff.transpose()) * R;
    state.cov = 0.5 * (state.cov + state.cov.transpose());
    if (correction_norm) *correction_norm = dx.norm();
    return dx.allFinite() && state.cov.allFinite();
  }

  // ponytail: fallback only when covariance math is ill-conditioned; upgrade path is a full batch pseudo update.
  dx = -gain * residual * H.transpose() / H_norm2;
  if (!dx.allFinite()) return false;
  state += dx;
  for (int i = 0; i < DIM_STATE; ++i)
  {
    if (std::fabs(H(0, i)) > 1e-9) state.cov(i, i) = std::min(state.cov(i, i), R);
  }
  state.cov = 0.5 * (state.cov + state.cov.transpose());
  if (correction_norm) *correction_norm = dx.norm();
  ROS_WARN_THROTTLE(1.0, "[DEGEN_GUARD] scalar pseudo measurement used temporary/simple correction fallback.");
  return dx.allFinite() && state.cov.allFinite();
}

void LIVMapper::applyDegeneracyGuardCorrections(const char *stage)
{
  if (!deg_guard_enable_) return;
  if (p_imu && p_imu->imu_need_init) return;
  if (gravity_align_en && !gravity_align_finished) return;

  deg_guard_last_z_correction_norm_ = 0.0;
  deg_guard_last_nhc_correction_norm_ = 0.0;
  deg_guard_last_velocity_body_ = _state.rot_end.transpose() * _state.vel_end;

  if (deg_guard_enable_z_soft_constraint_)
  {
    const double z_before = _state.pos_end[2];
    const double vz_before = _state.vel_end[2];
    deg_guard_last_z_residual_ = z_before - deg_guard_z_ref_;

    Eigen::Matrix<double, 1, DIM_STATE> H = Eigen::Matrix<double, 1, DIM_STATE>::Zero();
    double correction_norm = 0.0;
    H(0, 5) = 1.0;
    applyScalarPseudoMeasurement(_state, H, deg_guard_last_z_residual_, deg_guard_sigma_z_, deg_guard_z_gain_, &correction_norm);
    deg_guard_last_z_correction_norm_ += correction_norm;

    H.setZero();
    H(0, 9) = 1.0;
    applyScalarPseudoMeasurement(_state, H, _state.vel_end[2], deg_guard_sigma_vz_, deg_guard_z_gain_, &correction_norm);
    deg_guard_last_z_correction_norm_ += correction_norm;

    ROS_INFO_THROTTLE(1.0,
                      "[DEGEN_GUARD] z soft constraint stage=%s enabled=1 z=%.6f z_ref=%.6f r_z=%.6f v_z=%.6f corr_norm=%.6f scalar_EKF=1",
                      stage ? stage : "state",
                      z_before,
                      deg_guard_z_ref_,
                      deg_guard_last_z_residual_,
                      vz_before,
                      deg_guard_last_z_correction_norm_);
  }

  const bool nhc_mode_ok = !deg_guard_nhc_only_in_degenerate_ || deg_guard_corridor_degenerate_;
  const bool nhc_speed_ok = (_state.vel_end.norm() >= deg_guard_nhc_min_speed_) || deg_guard_corridor_degenerate_;
  if (deg_guard_enable_nhc_ && nhc_mode_ok && nhc_speed_ok)
  {
    const M3D R_world_to_body = _state.rot_end.transpose();
    V3D v_body = R_world_to_body * _state.vel_end;
    const double r_vy = v_body[1];
    const double r_vz = v_body[2];
    double correction_norm = 0.0;

    Eigen::Matrix<double, 1, DIM_STATE> H = Eigen::Matrix<double, 1, DIM_STATE>::Zero();
    for (int i = 0; i < 3; ++i) H(0, 7 + i) = R_world_to_body(1, i);
    applyScalarPseudoMeasurement(_state, H, r_vy, deg_guard_sigma_body_vy_, deg_guard_nhc_gain_, &correction_norm);
    deg_guard_last_nhc_correction_norm_ += correction_norm;

    const M3D R_world_to_body_after = _state.rot_end.transpose();
    v_body = R_world_to_body_after * _state.vel_end;
    H.setZero();
    for (int i = 0; i < 3; ++i) H(0, 7 + i) = R_world_to_body_after(2, i);
    applyScalarPseudoMeasurement(_state, H, v_body[2], deg_guard_sigma_body_vz_, deg_guard_nhc_gain_, &correction_norm);
    deg_guard_last_nhc_correction_norm_ += correction_norm;

    deg_guard_last_velocity_body_ = _state.rot_end.transpose() * _state.vel_end;
    ROS_INFO_THROTTLE(1.0,
                      "[DEGEN_GUARD] NHC stage=%s degenerate=%d v_world=[%.4f %.4f %.4f] v_body=[%.4f %.4f %.4f] r_vy=%.6f r_vz=%.6f corr_norm=%.6f",
                      stage ? stage : "state",
                      static_cast<int>(deg_guard_corridor_degenerate_),
                      _state.vel_end[0], _state.vel_end[1], _state.vel_end[2],
                      deg_guard_last_velocity_body_[0], deg_guard_last_velocity_body_[1], deg_guard_last_velocity_body_[2],
                      r_vy,
                      r_vz,
                      deg_guard_last_nhc_correction_norm_);
  }
  else
  {
    deg_guard_last_velocity_body_ = _state.rot_end.transpose() * _state.vel_end;
  }
}

void LIVMapper::evaluateDegeneracyGuardUpdate(const char *stage,
                                              const StatesGroup &state_before_update,
                                              int lio_feature_count,
                                              int visual_tracked_points)
{
  if (!deg_guard_enable_) return;

  deg_guard_last_update_status_ = "accepted";
  deg_guard_last_reject_reason_.clear();
  const std::string stage_name = stage ? stage : "state";
  const StatesGroup state_after_update = _state;
  const bool is_lio_update = stage_name.find("LIO") != std::string::npos || stage_name.find("LO") != std::string::npos;
  const bool is_vio_update = stage_name.find("VIO") != std::string::npos;
  deg_guard_last_sensor_type_ = is_lio_update ? "LIO" : is_vio_update ? "VIO" : stage_name;
  deg_guard_last_vio_skip_affects_degenerate_ = is_vio_update && deg_guard_use_vio_skip_for_degenerate_;

  const V3D update_translation = state_after_update.pos_end - state_before_update.pos_end;
  const Eigen::Matrix3d delta_rot = state_before_update.rot_end.transpose() * state_after_update.rot_end;
  const double yaw_rad = std::atan2(delta_rot(1, 0), delta_rot(0, 0));
  deg_guard_last_update_translation_norm_ = update_translation.norm();
  deg_guard_last_update_yaw_deg_ = std::fabs(yaw_rad) * 57.29577951308232;
  const double current_time = LidarMeasures.last_lio_update_time;
  double *last_sensor_time = is_vio_update ? &deg_guard_last_vio_time_ : &deg_guard_last_lio_time_;
  const double fallback_dt = is_vio_update ? deg_guard_camera_dt_ : deg_guard_lidar_dt_;
  if (last_sensor_time && *last_sensor_time >= 0.0 && current_time > *last_sensor_time)
  {
    deg_guard_last_dt_ = current_time - *last_sensor_time;
  }
  else
  {
    deg_guard_last_dt_ = fallback_dt;
  }
  if (last_sensor_time) *last_sensor_time = current_time;
  deg_guard_last_dt_ = std::max(1e-4, deg_guard_last_dt_);
  deg_guard_last_update_translation_rate_mps_ = deg_guard_last_update_translation_norm_ / deg_guard_last_dt_;
  deg_guard_last_update_yaw_rate_degps_ = deg_guard_last_update_yaw_deg_ / deg_guard_last_dt_;
  deg_guard_last_final_pose_delta_ = deg_guard_last_update_translation_norm_;
  if (is_vio_update && vio_manager)
  {
    deg_guard_last_visual_update_rot_deg_ = vio_manager->last_visual_update_rot_deg;
    deg_guard_last_visual_update_rot_rate_degps_ = vio_manager->last_visual_update_rot_rate_degps;
  }
  else
  {
    deg_guard_last_visual_update_rot_deg_ = 0.0;
    deg_guard_last_visual_update_rot_rate_degps_ = 0.0;
  }
  deg_guard_last_lio_feature_count_ = lio_feature_count;
  deg_guard_last_visual_tracked_points_ = visual_tracked_points;
  deg_guard_last_velocity_body_ = state_after_update.rot_end.transpose() * state_after_update.vel_end;

  V3D forward_axis = state_after_update.rot_end.col(0);
  if (!forward_axis.allFinite() || forward_axis.norm() < 1e-9) forward_axis = V3D::UnitX();
  else forward_axis.normalize();
  deg_guard_last_forward_progress_ = 0.0;
  if (deg_guard_last_pos_ready_)
  {
    deg_guard_last_forward_progress_ = (state_after_update.pos_end - deg_guard_last_pos_).dot(forward_axis);
  }
  deg_guard_last_forward_progress_rate_mps_ = deg_guard_last_forward_progress_ / deg_guard_last_dt_;

  const bool backward_by_step = deg_guard_last_pos_ready_ &&
                                deg_guard_last_forward_progress_rate_mps_ < -deg_guard_backward_speed_threshold_;
  // ponytail: v_body_x is diagnostic only here; do not reject/downweight an update only because forward velocity is negative.
  const bool backward_candidate = deg_guard_enable_backward_guard_ && backward_by_step;
  deg_guard_backward_count_ = backward_candidate ? (deg_guard_backward_count_ + 1) : 0;
  deg_guard_last_backward_slip_ = deg_guard_backward_count_ >= deg_guard_backward_consecutive_frames_;

  bool suspicious = false;
  std::ostringstream reason;
  bool has_reason = false;
  auto add_reason = [&](const std::string &item) {
    if (has_reason) reason << ";";
    reason << item;
    has_reason = true;
  };

  const bool update_finite = state_after_update.pos_end.allFinite() &&
                             state_after_update.vel_end.allFinite() &&
                             state_after_update.rot_end.allFinite();
  if (!update_finite)
  {
    suspicious = true;
    add_reason("nonfinite_update");
  }
  if (deg_guard_enable_corridor_detection_)
  {
    if (is_lio_update && lio_feature_count >= 0 && lio_feature_count < deg_guard_min_lidar_features_)
    {
      suspicious = true;
      add_reason("low_lidar_features");
    }
    if (is_lio_update && voxelmap_manager && voxelmap_manager->isLidarDegenerated())
    {
      suspicious = true;
      add_reason("lidar_constraint_degenerated");
    }
    if (is_vio_update && deg_guard_use_vio_skip_for_degenerate_ &&
        visual_tracked_points >= 0 && visual_tracked_points < deg_guard_min_visual_tracked_points_)
    {
      suspicious = true;
      add_reason("low_visual_points");
    }
    if (deg_guard_max_update_translation_rate_mps_ > 0.0 &&
        deg_guard_last_update_translation_rate_mps_ > deg_guard_max_update_translation_rate_mps_)
    {
      suspicious = true;
      add_reason("large_update_translation_rate");
    }
    if (deg_guard_max_update_yaw_rate_degps_ > 0.0 &&
        deg_guard_last_update_yaw_rate_degps_ > deg_guard_max_update_yaw_rate_degps_)
    {
      if (!is_vio_update || deg_guard_use_vio_large_rotation_for_reject_)
      {
        suspicious = true;
        add_reason("large_update_yaw_rate");
      }
    }
  }

  if (suspicious)
  {
    deg_guard_degenerate_count_++;
    deg_guard_recover_count_ = 0;
  }
  else
  {
    deg_guard_recover_count_++;
    deg_guard_degenerate_count_ = 0;
  }

  const std::string reason_text = has_reason ? reason.str() : "normal";
  if (!deg_guard_corridor_degenerate_ && deg_guard_degenerate_count_ >= deg_guard_min_degenerate_frames_)
  {
    deg_guard_corridor_degenerate_ = true;
    ROS_WARN("[DEGEN_GUARD] enter corridor degeneracy mode reason=%s", reason_text.c_str());
  }
  else if (deg_guard_corridor_degenerate_ && deg_guard_recover_count_ >= deg_guard_recover_frames_)
  {
    deg_guard_corridor_degenerate_ = false;
    ROS_WARN("[DEGEN_GUARD] exit corridor degeneracy mode reason=%s", reason_text.c_str());
  }

  std::string action = "none";
  if (deg_guard_last_backward_slip_)
  {
    action = deg_guard_backward_action_;
  }

  const bool large_degenerate_update =
      deg_guard_corridor_degenerate_ &&
      ((deg_guard_max_degenerate_update_translation_ > 0.0 &&
        deg_guard_last_update_translation_norm_ > deg_guard_max_degenerate_update_translation_) ||
       (deg_guard_max_degenerate_update_yaw_deg_ > 0.0 &&
        deg_guard_last_update_yaw_deg_ > deg_guard_max_degenerate_update_yaw_deg_));
  if (deg_guard_reject_large_update_in_degenerate_ && large_degenerate_update)
  {
    action = "reject_update";
  }
  if (!update_finite && deg_guard_reject_nonfinite_update_)
  {
    action = "reject_update";
  }
  else if (deg_guard_corridor_degenerate_ && action == "none")
  {
    action = "downweight_update";
  }

  deg_guard_last_action_ = action;
  deg_guard_last_reason_ = reason_text;
  if (action == "log_only")
  {
    deg_guard_last_update_status_ = "accepted";
  }

  if (deg_guard_last_backward_slip_)
  {
    ROS_WARN_THROTTLE(0.5,
                      "[DEGEN_GUARD] backward slip t=%.6f forward_progress=%.6f v_body_x=%.6f count=%d action=%s degenerate=%d",
                      LidarMeasures.last_lio_update_time,
                      deg_guard_last_forward_progress_,
                      deg_guard_last_velocity_body_[0],
                      deg_guard_backward_count_,
                      action.c_str(),
                      static_cast<int>(deg_guard_corridor_degenerate_));
  }

  if (action == "reject_update")
  {
    deg_guard_last_update_status_ = "rejected";
    deg_guard_last_reject_reason_ = reason_text;
    _state = state_before_update;
    ROS_WARN_THROTTLE(0.5,
                      "[DEGEN_GUARD] reject %s update: dtrans=%.4f m dyaw=%.4f deg reason=%s",
                      stage_name.c_str(),
                      deg_guard_last_update_translation_norm_,
                      deg_guard_last_update_yaw_deg_,
                      reason_text.c_str());
  }
  else if (action == "downweight_update")
  {
    deg_guard_last_update_status_ = "downweighted";
    const double noise_scale = is_lio_update ? deg_guard_degenerate_lio_noise_scale_ : deg_guard_degenerate_vio_noise_scale_;
    const double update_scale = 1.0 / std::max(1.0, noise_scale);
    blendDegeneracyGuardUpdate(state_before_update, state_after_update, update_scale);
    ROS_WARN_THROTTLE(0.5,
                      "[DEGEN_GUARD] downweight %s update: scale=%.3f dtrans=%.4f m dyaw=%.4f deg reason=%s",
                      stage_name.c_str(),
                      update_scale,
                      deg_guard_last_update_translation_norm_,
                      deg_guard_last_update_yaw_deg_,
                      reason_text.c_str());
  }
  else if (action == "log_only")
  {
    ROS_WARN_THROTTLE(0.5,
                      "[DEGEN_GUARD] log-only %s update: dtrans=%.4f m dyaw=%.4f deg reason=%s",
                      stage_name.c_str(),
                      deg_guard_last_update_translation_norm_,
                      deg_guard_last_update_yaw_deg_,
                      reason_text.c_str());
  }

  deg_guard_last_pos_ = _state.pos_end;
  deg_guard_last_pos_ready_ = _state.pos_end.allFinite();
  deg_guard_last_velocity_body_ = _state.rot_end.transpose() * _state.vel_end;
}

void LIVMapper::writeDegeneracyGuardLog(const char *stage)
{
		  if (diagnostics_level_ != "verbose" ||
		      !(deg_guard_enable_ || safety_guard_enable_ || corridor_prior_enable_) ||
		      !degeneracy_guard_log_.is_open()) return;
  const double now = LidarMeasures.last_lio_update_time;
  local_elapsed_sec_ = (_first_lidar_time > 0.0 && now >= _first_lidar_time) ? now - _first_lidar_time : 0.0;
  bag_elapsed_sec_ = bag_start_offset_ + local_elapsed_sec_;
  if (debug_fixed_degraded_intervals_enable_ && fixed_degraded_trigger_mode_ == "manual_time")
  {
    const bool in_first =
        bag_elapsed_sec_ >= fixed_degraded_first_start_sec_ &&
        bag_elapsed_sec_ <= fixed_degraded_first_end_sec_;
    const bool in_second =
        bag_elapsed_sec_ >= fixed_degraded_second_start_sec_ &&
        bag_elapsed_sec_ <= fixed_degraded_second_end_sec_;
    in_fixed_degraded_window_ = in_first || in_second;
    fixed_degraded_reason_ = in_first ? "debug_fixed_degraded_window_1" :
                       in_second ? "debug_fixed_degraded_window_2" : "not_in_window";
  }
  else
  {
    in_fixed_degraded_window_ = false;
    fixed_degraded_reason_ = "disabled";
  }
	  const V3D v_body = _state.rot_end.transpose() * _state.vel_end;
	  const std::string mode = update_decision_ready_ ?
	                           systemModeName(current_decision_.mode) :
	                           (safety_fail_safe_mode_ ? "DEGRADED_HOLD" :
	                            (deg_guard_corridor_degenerate_ ? "DEGRADED_BOOTSTRAP" : "NORMAL"));
	  const UwbUpdateSummary empty_uwb_summary;
	  const UwbUpdateSummary &uwb_summary = uwb_manager ? uwb_manager->lastUpdateSummary() : empty_uwb_summary;
	  degeneracy_guard_log_ << std::fixed << std::setprecision(9)
	                        << LidarMeasures.last_lio_update_time << ","
	                        << deterministic_frame_id_ << ","
	                        << (stage ? stage : "state") << ","
	                        << deg_guard_last_sensor_type_ << ","
		                        << mode << ","
		                        << local_mode_ << ","
		                        << static_cast<int>(deterministic_mode_) << ","
		                        << deg_guard_last_dt_ << ","
		                        << local_elapsed_sec_ << ","
		                        << bag_start_offset_ << ","
		                        << bag_elapsed_sec_ << ","
		                        << static_cast<int>(in_fixed_degraded_window_) << ","
		                        << fixed_degraded_reason_ << ","
		                        << static_cast<int>(deterministic_last_lio_update_) << ","
	                        << static_cast<int>(deterministic_last_vio_update_) << ","
	                        << static_cast<int>(deterministic_last_uwb_update_) << ","
	                        << deterministic_last_uwb_anchor_ids_ << ","
	                        << uwb_summary.xy_correction_before_step << ","
	                        << uwb_summary.z_correction_before_clamp << ","
	                        << uwb_summary.z_correction_after_clamp << ","
	                        << static_cast<int>(local_map_cleared_last_) << ","
	                        << static_cast<int>(visual_map_cleared_last_) << ","
	                        << static_cast<int>(tracker_reset_last_) << ","
	                        << lio_bootstrap_frames_ << ","
	                        << vio_bootstrap_frames_ << ","
	                        << static_cast<int>(local_vio_unavailable_) << ","
	                        << static_cast<int>(local_lio_weak_) << ","
	                        << _state.pos_end[0] << "," << _state.pos_end[1] << "," << _state.pos_end[2] << ","
                        << _state.vel_end[0] << "," << _state.vel_end[1] << "," << _state.vel_end[2] << ","
                        << v_body[0] << "," << v_body[1] << "," << v_body[2] << ","
                        << safety_last_speed_ << ","
                        << safety_last_quat_norm_ << ","
	                        << safety_last_frame_translation_ << ","
	                        << safety_last_frame_rotation_deg_ << ","
	                        << safety_backward_distance_in_window_ << ","
	                        << corridor_prior_axis_[0] << "," << corridor_prior_axis_[1] << "," << corridor_prior_axis_[2] << ","
	                        << corridor_prior_progress_ << ","
	                        << corridor_prior_progress_delta_1s_ << ","
	                        << corridor_prior_progress_delta_window_ << ","
	                        << corridor_prior_backward_distance_window_ << ","
	                        << corridor_prior_backward_distance_fail_window_ << ","
	                        << corridor_prior_action_ << ","
	                        << static_cast<int>(corridor_prior_update_voxel_map_enabled_) << ","
	                        << static_cast<int>(corridor_prior_visual_map_update_enabled_) << ","
	                        << _state.pos_end[2] << ","
                        << deg_guard_last_z_residual_ << ","
                        << deg_guard_last_forward_progress_ << ","
                        << static_cast<int>(deg_guard_last_backward_slip_) << ","
                        << deg_guard_last_lio_feature_count_ << ","
                        << deg_guard_last_visual_tracked_points_ << ","
                        << deg_guard_last_update_translation_norm_ << ","
                        << deg_guard_last_update_translation_rate_mps_ << ","
                        << deg_guard_last_update_yaw_deg_ << ","
                        << deg_guard_last_update_yaw_rate_degps_ << ","
                        << deg_guard_last_visual_update_rot_deg_ << ","
                        << deg_guard_last_visual_update_rot_rate_degps_ << ","
                        << deg_guard_last_z_correction_norm_ << ","
                        << deg_guard_last_nhc_correction_norm_ << ","
                        << deg_guard_last_lio_noise_scale_ << ","
                        << deg_guard_last_vio_noise_scale_ << ","
                        << deg_guard_last_weight_reason_ << ","
                        << static_cast<int>(deg_guard_last_lio_update_executed_) << ","
                        << static_cast<int>(deg_guard_last_lio_downweighted_) << ","
                        << static_cast<int>(deg_guard_last_lio_voxel_map_updated_) << ","
                        << deg_guard_last_lio_voxel_map_skip_reason_ << ","
                        << static_cast<int>(deg_guard_last_vio_skip_affects_degenerate_) << ","
                        << deg_guard_last_update_status_ << ","
                        << deg_guard_last_reject_reason_ << ","
                        << deg_guard_last_action_ << ","
                        << (safety_fail_safe_mode_ ? safety_last_reason_ : deg_guard_last_reason_) << ","
                        << deg_guard_last_final_pose_delta_ << "\n";
  degeneracy_guard_log_.flush();
}

bool LIVMapper::isStateFiniteForSafety(const StatesGroup &state, const char *stage, std::string *reason) const
{
  if (!safety_guard_enable_) return true;
  auto setReason = [&](const std::string &why) {
    if (reason != nullptr) *reason = std::string(stage ? stage : "state") + ":" + why;
  };

  if (!state.pos_end.allFinite())
  {
    setReason("nonfinite_position");
    return false;
  }
  if (!state.vel_end.allFinite())
  {
    setReason("nonfinite_velocity");
    return false;
  }
  if (!state.rot_end.allFinite())
  {
    setReason("nonfinite_rotation");
    return false;
  }
  if (!state.bias_g.allFinite() || !state.bias_a.allFinite())
  {
    setReason("nonfinite_imu_bias");
    return false;
  }
  if (!state.gravity.allFinite())
  {
    setReason("nonfinite_gravity");
    return false;
  }
  if (!std::isfinite(state.inv_expo_time))
  {
    setReason("nonfinite_inv_exposure");
    return false;
  }
  if (state.cov.rows() != DIM_STATE || state.cov.cols() != DIM_STATE || !state.cov.allFinite())
  {
    setReason("nonfinite_covariance");
    return false;
  }
  const Eigen::Quaterniond q(state.rot_end);
  const double q_norm = q.norm();
  if (!std::isfinite(q_norm) || q_norm < 1e-6)
  {
    setReason("invalid_quaternion_norm");
    return false;
  }
  return true;
}

bool LIVMapper::validateStateForSafety(const char *stage,
                                       const StatesGroup &state_before,
                                       const StatesGroup &state_after,
                                       bool update_backward_window,
                                       bool allow_recover)
{
  if (!safety_guard_enable_) return true;

  std::string reason;
  if (!isStateFiniteForSafety(state_after, stage, &reason))
  {
    enterFailSafe(reason, &state_before);
    return false;
  }

  safety_last_speed_ = state_after.vel_end.norm();
  safety_last_quat_norm_ = Eigen::Quaterniond(state_after.rot_end).norm();
  safety_last_frame_translation_ = (state_after.pos_end - state_before.pos_end).norm();
  safety_last_frame_rotation_deg_ = 0.0;
  if (state_before.rot_end.allFinite())
  {
    const Eigen::Matrix3d delta_rot = state_before.rot_end.transpose() * state_after.rot_end;
    safety_last_frame_rotation_deg_ = Eigen::AngleAxisd(delta_rot).angle() * 57.29577951308232;
  }

  // ponytail: startup gravity alignment is a legitimate one-shot pose jump; arm motion sanity after one reliable state.
  if (!safety_reliable_state_ready_)
  {
    safety_last_reason_ = "arming";
    return true;
  }

  // ponytail: local/bootstrap rebuilds intentionally change map support; keep the crash guard, skip motion sanity.
  if (isBootstrapMode())
  {
    safety_backward_window_.clear();
    safety_backward_distance_in_window_ = 0.0;
    if (allow_recover) maybeRecoverFailSafe();
    safety_last_reason_ = safety_fail_safe_mode_ ? safety_last_reason_ : "bootstrap_relaxed";
    return true;
  }

  const double now = LidarMeasures.last_lio_update_time;
  if (update_backward_window && std::isfinite(now))
  {
    Eigen::Vector3d forward_axis = state_after.rot_end.col(0);
    if (forward_axis.allFinite() && forward_axis.norm() > 1e-6)
    {
      forward_axis.normalize();
      const double forward_progress = (state_after.pos_end - state_before.pos_end).dot(forward_axis);
      const double backward_step = std::max(0.0, -forward_progress);
      if (backward_step > 0.0) safety_backward_window_.push_back({now, backward_step});
    }
    while (!safety_backward_window_.empty() &&
           now - safety_backward_window_.front().first > safety_backward_time_window_)
    {
      safety_backward_window_.pop_front();
    }
    safety_backward_distance_in_window_ = 0.0;
    for (const auto &sample : safety_backward_window_)
    {
      safety_backward_distance_in_window_ += sample.second;
    }
  }

  if (safety_max_speed_ > 0.0 && safety_last_speed_ > safety_max_speed_)
  {
    enterFailSafe("speed_over_limit", &state_before);
    return false;
  }
  if (safety_max_frame_translation_ > 0.0 &&
      safety_last_frame_translation_ > safety_max_frame_translation_)
  {
    enterFailSafe("frame_translation_over_limit", &state_before);
    return false;
  }
  if (safety_max_frame_rotation_deg_ > 0.0 &&
      safety_last_frame_rotation_deg_ > safety_max_frame_rotation_deg_)
  {
    enterFailSafe("frame_rotation_over_limit", &state_before);
    return false;
  }
  if (safety_backward_distance_threshold_ > 0.0 &&
      safety_backward_distance_in_window_ > safety_backward_distance_threshold_)
  {
    const std::string backward_reason = "backward_window_over_limit";
    if (safety_backward_action_ == "reject_update" ||
        safety_backward_action_ == "fail_safe_or_downweight")
    {
      enterFailSafe(backward_reason, &state_before);
      return false;
    }
    if (safety_backward_action_ == "downweight_update")
    {
      deg_guard_last_update_status_ = "downweighted";
      deg_guard_last_reject_reason_ = backward_reason;
      deg_guard_last_action_ = safety_backward_action_;
      deg_guard_last_reason_ = backward_reason;
    }
  }

  if (allow_recover) maybeRecoverFailSafe();
  safety_last_reason_ = safety_fail_safe_mode_ ? safety_last_reason_ : "normal";
  return true;
}

void LIVMapper::enterFailSafe(const std::string &reason, const StatesGroup *fallback_state)
{
  if (!safety_guard_enable_) return;

  const bool was_fail_safe = safety_fail_safe_mode_;
  safety_fail_safe_mode_ = true;
  safety_fail_safe_stable_frames_ = 0;
  safety_last_reason_ = reason;
  deg_guard_last_update_status_ = "rejected";
  deg_guard_last_reject_reason_ = reason;
  deg_guard_last_action_ = "fail_safe";
  deg_guard_last_reason_ = reason;
  deg_guard_last_lio_update_executed_ = false;
  deg_guard_last_lio_voxel_map_updated_ = false;
  deg_guard_last_lio_voxel_map_skip_reason_ = "DEGRADED_HOLD";

  if (safety_reliable_state_ready_)
  {
    _state = safety_reliable_state_;
  }
  else if (fallback_state != nullptr && isStateFiniteForSafety(*fallback_state, "fail_safe_fallback", nullptr))
  {
    _state = *fallback_state;
  }
  else
  {
    _state = StatesGroup();
  }
  _state.vel_end.setZero();
  state_propagat = _state;
  if (voxelmap_manager) voxelmap_manager->state_ = _state;
  if (vio_manager) vio_manager->updateFrameState(_state);

  clearSafetyLocalCaches();

  if (!was_fail_safe)
  {
    ROS_ERROR("[FATAL_STATE] Enter FAIL_SAFE: reason=%s, hold_pos=(%.3f %.3f %.3f), "
              "speed=%.3f, frame_trans=%.3f, frame_rot=%.3f, backward_window=%.3f.",
              reason.c_str(),
              _state.pos_end[0], _state.pos_end[1], _state.pos_end[2],
              safety_last_speed_,
              safety_last_frame_translation_,
              safety_last_frame_rotation_deg_,
              safety_backward_distance_in_window_);
  }
}

void LIVMapper::maybeRecoverFailSafe()
{
  if (!safety_guard_enable_ || !safety_fail_safe_mode_) return;
  std::string reason;
  const bool stable = isStateFiniteForSafety(_state, "fail_safe_recover", &reason) &&
                      (safety_max_speed_ <= 0.0 || _state.vel_end.norm() <= 0.5 * safety_max_speed_);
  safety_fail_safe_stable_frames_ = stable ? (safety_fail_safe_stable_frames_ + 1) : 0;
	  if (safety_fail_safe_stable_frames_ >= safety_fail_safe_recover_frames_)
	  {
	    safety_fail_safe_mode_ = false;
	    safety_last_reason_ = "recover_local_reinit";
	    safety_backward_window_.clear();
	    safety_backward_distance_in_window_ = 0.0;
	    const StatesGroup recovered_state = _state;
	    if (safety_reliable_state_ready_) _state = safety_reliable_state_;
	    else safety_reliable_state_ = recovered_state;
	    safety_reliable_state_ready_ = true;
	    ROS_WARN("[FATAL_STATE] Exit FAIL_SAFE after %d stable frames: last_good_pose=[%.3f %.3f %.3f] rejected_pose=[%.3f %.3f %.3f] recovered_pose=[%.3f %.3f %.3f] mode_after_recovery=LOCAL_REINIT",
	             safety_fail_safe_stable_frames_,
	             safety_reliable_state_.pos_end.x(), safety_reliable_state_.pos_end.y(), safety_reliable_state_.pos_end.z(),
	             recovered_state.pos_end.x(), recovered_state.pos_end.y(), recovered_state.pos_end.z(),
	             _state.pos_end.x(), _state.pos_end.y(), _state.pos_end.z());
	    safety_fail_safe_stable_frames_ = 0;
	    if (local_reinit_enable_) enterLocalReinitMode("LOCAL_REINIT", "safety_recover_local_reinit");
	  }
	}

void LIVMapper::recordReliableStateForSafety(const char *stage)
{
  if (!safety_guard_enable_ || safety_fail_safe_mode_) return;
  std::string reason;
  if (!isStateFiniteForSafety(_state, stage, &reason)) return;
  const double speed = _state.vel_end.norm();
  if (safety_max_speed_ > 0.0 && speed > safety_max_speed_) return;
  safety_reliable_state_ = _state;
  safety_reliable_state_ready_ = true;
  safety_last_speed_ = speed;
  safety_last_quat_norm_ = Eigen::Quaterniond(_state.rot_end).norm();
}

void LIVMapper::clearSafetyLocalCaches()
{
  _pv_list.clear();
  if (voxelmap_manager)
  {
    voxelmap_manager->pv_list_.clear();
    voxelmap_manager->ptpl_list_.clear();
    voxelmap_manager->cross_mat_list_.clear();
    voxelmap_manager->body_cov_list_.clear();
  }
  if (vio_manager && vio_manager->visual_submap != nullptr)
  {
    vio_manager->visual_submap->reset();
  }
}

void LIVMapper::clearLocalMapsForReinit(const std::string &reason)
{
  if (!experimental_features_enable_) return;
  local_map_cleared_last_ = false;
  visual_map_cleared_last_ = false;
  tracker_reset_last_ = false;

  std::unordered_set<VoxelOctoTree *> deleted_voxels;
  auto deleteVoxelMap = [&](std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &map) {
    for (auto &kv : map)
    {
      if (kv.second != nullptr && deleted_voxels.insert(kv.second).second) delete kv.second;
    }
    map.clear();
  };

  deleteVoxelMap(voxel_map);
  if (voxelmap_manager)
  {
    deleteVoxelMap(voxelmap_manager->voxel_map_);
    deleteVoxelMap(voxelmap_manager->long_term_visual_map_);
    voxelmap_manager->visual_observed_voxels_.clear();
    voxelmap_manager->pv_list_.clear();
    voxelmap_manager->ptpl_list_.clear();
    voxelmap_manager->cross_mat_list_.clear();
    voxelmap_manager->body_cov_list_.clear();
    voxelmap_manager->lidar_degenerated_ = false;
    voxelmap_manager->lidar_constraint_ratio_ = 0.0;
  }
  _pv_list.clear();
  lidar_map_inited = false;
  lio_frames_since_voxel_map_update_ = 0;
  lio_last_voxel_map_update_time_ = -1.0;
  local_map_cleared_last_ = true;

  if (vio_manager)
  {
    if (vio_manager->visual_submap != nullptr) vio_manager->visual_submap->reset();
    for (auto &kv : vio_manager->feat_map) delete kv.second;
    vio_manager->feat_map.clear();
    vio_manager->sub_feat_map.clear();
    vio_manager->resetGrid();
    vio_manager->total_points = 0;
    visual_map_cleared_last_ = true;
    tracker_reset_last_ = true;
  }

  _state.cov.block<3, 3>(3, 3) += M3D::Identity() * 0.25;
  _state.cov.block<3, 3>(6, 6) += M3D::Identity() * 0.25;
  _state.cov = 0.5 * (_state.cov + _state.cov.transpose());
  state_propagat = _state;
  if (voxelmap_manager) voxelmap_manager->state_ = _state;
  if (vio_manager) vio_manager->updateFrameState(_state);

  ROS_WARN("[LOCAL_REINIT] Cleared local maps: reason=%s mode=%s keep_pose=[%.3f %.3f %.3f]",
           reason.c_str(), local_mode_.c_str(),
           _state.pos_end.x(), _state.pos_end.y(), _state.pos_end.z());
}

void LIVMapper::enterLocalReinitMode(const std::string &mode, const std::string &reason)
{
  if (!experimental_features_enable_) return;
  if (!local_reinit_enable_) return;
  if (local_mode_ == mode && local_reinit_reason_ == reason) return;
  local_mode_ = mode;
  local_reinit_reason_ = reason;
  local_mode_start_time_ = LidarMeasures.last_lio_update_time;
  lio_bootstrap_frames_ = 0;
  vio_bootstrap_frames_ = 0;
  local_map_cleared_last_ = false;
  visual_map_cleared_last_ = false;
  tracker_reset_last_ = false;
  if (mode == "DEGRADED_HOLD")
  {
    degraded_hold_entry_pos_ = _state.pos_end;
    degraded_hold_entry_state_ = _state;
    degraded_hold_entry_state_.vel_end.setZero();
    degraded_hold_entry_state_ready_ = true;
    degraded_hold_entry_rpy_ = RotMtoEuler(_state.rot_end);
    degraded_hold_last_reject_reason_ = "none";
    _state = degraded_hold_entry_state_;
    state_propagat = _state;
    if (voxelmap_manager) voxelmap_manager->state_ = _state;
    if (vio_manager) vio_manager->updateFrameState(_state);
    if (safety_guard_enable_)
    {
      safety_reliable_state_ = _state;
      safety_reliable_state_ready_ = true;
      safety_backward_window_.clear();
      safety_backward_distance_in_window_ = 0.0;
      safety_last_speed_ = 0.0;
      safety_last_frame_translation_ = 0.0;
      safety_last_frame_rotation_deg_ = 0.0;
    }
    resetCorridorMotionPrior();
    ROS_WARN("[LOCAL_REINIT] Enter DEGRADED_HOLD: reason=%s entry_pose=[%.3f %.3f %.3f] entry_rpy_deg=[%.3f %.3f %.3f]",
             reason.c_str(),
             degraded_hold_entry_pos_.x(), degraded_hold_entry_pos_.y(), degraded_hold_entry_pos_.z(),
             degraded_hold_entry_rpy_.x() * 57.29577951308232,
             degraded_hold_entry_rpy_.y() * 57.29577951308232,
             degraded_hold_entry_rpy_.z() * 57.29577951308232);
    return;
  }

  if (mode == "DEGRADED_BOOTSTRAP" || mode == "LOCAL_REINIT" || mode == "RELOCALIZE")
  {
    if (mode == "LOCAL_REINIT" &&
        reason.find("BACKWARD") != std::string::npos &&
        safety_reliable_state_ready_)
    {
      _state = safety_reliable_state_;
    }
    _state.vel_end.setZero();
    resetCorridorMotionPrior();
    clearLocalMapsForReinit(reason);
    _state.vel_end.setZero();
    state_propagat = _state;
    if (voxelmap_manager) voxelmap_manager->state_ = _state;
    if (vio_manager) vio_manager->updateFrameState(_state);
    if (safety_guard_enable_)
    {
      safety_reliable_state_ = _state;
      safety_reliable_state_ready_ = true;
      safety_backward_window_.clear();
      safety_backward_distance_in_window_ = 0.0;
      safety_last_speed_ = 0.0;
      safety_last_frame_translation_ = 0.0;
      safety_last_frame_rotation_deg_ = 0.0;
      safety_last_reason_ = "local_reinit";
    }
  }
  ROS_WARN("[LOCAL_REINIT] Enter %s: reason=%s", mode.c_str(), reason.c_str());
}

bool LIVMapper::isDegradedHoldMode() const
{
  if (!experimental_features_enable_) return false;
  return local_reinit_enable_ && local_mode_ == "DEGRADED_HOLD";
}

bool LIVMapper::isBootstrapMode() const
{
  if (!experimental_features_enable_) return false;
  return local_reinit_enable_ &&
         (local_mode_ == "DEGRADED_BOOTSTRAP" ||
          local_mode_ == "LOCAL_REINIT" ||
          local_mode_ == "RELOCALIZE");
}

void LIVMapper::applyDegradedHoldConstraint(const char *stage, bool check_reject)
{
  if (!isDegradedHoldMode() || !degraded_hold_entry_state_ready_) return;

  const V3D rpy = RotMtoEuler(_state.rot_end);
  const double roll_err_deg = std::fabs(rpy.x() - degraded_hold_entry_rpy_.x()) * 57.29577951308232;
  const double pitch_err_deg = std::fabs(rpy.y() - degraded_hold_entry_rpy_.y()) * 57.29577951308232;
  const double speed = _state.vel_end.norm();
  degraded_hold_last_reject_reason_ = "none";
  bool reject = false;
  std::string reject_reason;
  if (check_reject && degraded_hold_attitude_reject_deg_ > 0.0 &&
      (roll_err_deg > degraded_hold_attitude_reject_deg_ || pitch_err_deg > degraded_hold_attitude_reject_deg_))
  {
    reject = true;
    reject_reason = "DEGRADED_ATTITUDE_REJECT";
  }
  if (check_reject && degraded_hold_speed_reject_mps_ > 0.0 && speed > degraded_hold_speed_reject_mps_)
  {
    reject = true;
    reject_reason = reject_reason.empty() ? "DEGRADED_SPEED_REJECT" : reject_reason + ";DEGRADED_SPEED_REJECT";
  }

  if (reject)
  {
    degraded_hold_last_reject_reason_ = reject_reason;
    ROS_WARN_THROTTLE(0.5,
                      "[LOCAL_REINIT] %s stage=%s roll_err=%.3f pitch_err=%.3f speed=%.3f entry_pose=[%.3f %.3f %.3f]",
                      reject_reason.c_str(),
                      stage ? stage : "state",
                      roll_err_deg,
                      pitch_err_deg,
                      speed,
                      degraded_hold_entry_state_.pos_end.x(),
                      degraded_hold_entry_state_.pos_end.y(),
                      degraded_hold_entry_state_.pos_end.z());
  }

  _state.pos_end = degraded_hold_entry_state_.pos_end;
  _state.rot_end = degraded_hold_entry_state_.rot_end;
  _state.vel_end.setZero();
  _state.cov.block<3, 3>(3, 3) =
      0.5 * (_state.cov.block<3, 3>(3, 3) + degraded_hold_entry_state_.cov.block<3, 3>(3, 3));
  for (int i = 7; i < 10; ++i) _state.cov(i, i) = std::min(_state.cov(i, i), 0.01);
  _state.cov = 0.5 * (_state.cov + _state.cov.transpose());
  state_propagat = _state;
  if (voxelmap_manager) voxelmap_manager->state_ = _state;
  if (vio_manager) vio_manager->updateFrameState(_state);
}

bool LIVMapper::localModeBlocksMapUpdate() const
{
  if (!experimental_features_enable_) return false;
  if (!local_reinit_enable_) return false;
  if (local_mode_ == "DEGRADED_HOLD") return disable_voxel_map_in_degraded_hold_;
  return local_mode_ == "RELOCALIZE";
}

bool LIVMapper::localModeBlocksVisualMapUpdate() const
{
  if (!experimental_features_enable_) return false;
  if (!local_reinit_enable_) return false;
  if (local_mode_ == "DEGRADED_HOLD") return disable_visual_map_in_degraded_hold_;
  return local_mode_ == "RELOCALIZE";
}

bool LIVMapper::localModeSkipsVisualEkf() const
{
  if (!experimental_features_enable_) return false;
  if (!local_reinit_enable_) return false;
  if (local_mode_ == "DEGRADED_HOLD") return true;
  if (local_mode_ == "DEGRADED_BOOTSTRAP") return degraded_bootstrap_enable_;
  return local_mode_ == "LOCAL_REINIT" || local_mode_ == "RELOCALIZE";
}

void LIVMapper::updateLocalModeAtFrameStart()
{
  local_map_cleared_last_ = false;
  visual_map_cleared_last_ = false;
  tracker_reset_last_ = false;
  deterministic_last_lio_update_ = false;
  deterministic_last_vio_update_ = false;
  deterministic_last_uwb_update_ = false;
  deterministic_last_uwb_anchor_ids_.clear();

  const double now = LidarMeasures.last_lio_update_time;
  local_elapsed_sec_ = (_first_lidar_time > 0.0 && now >= _first_lidar_time) ? now - _first_lidar_time : 0.0;
  bag_elapsed_sec_ = bag_start_offset_ + local_elapsed_sec_;
  in_fixed_degraded_window_ = false;
  fixed_degraded_reason_ = debug_fixed_degraded_intervals_enable_ ? "not_in_window" : "disabled";

  if (!experimental_features_enable_)
  {
    local_mode_ = "NORMAL";
    local_reinit_reason_ = "experimental_disabled";
    return;
  }

  if (!local_reinit_enable_)
  {
    local_mode_ = "NORMAL";
    local_reinit_reason_ = "disabled";
    return;
  }

  if (debug_fixed_degraded_intervals_enable_)
  {
    const bool in_first =
        bag_elapsed_sec_ >= fixed_degraded_first_start_sec_ &&
        bag_elapsed_sec_ <= fixed_degraded_first_end_sec_;
    const bool in_second =
        bag_elapsed_sec_ >= fixed_degraded_second_start_sec_ &&
        bag_elapsed_sec_ <= fixed_degraded_second_end_sec_;
    in_fixed_degraded_window_ = in_first || in_second;
    fixed_degraded_reason_ = in_first ? "debug_fixed_degraded_window_1" :
                       in_second ? "debug_fixed_degraded_window_2" : "not_in_window";

    if (in_fixed_degraded_window_)
    {
      if (local_mode_ != "DEGRADED_HOLD") enterLocalReinitMode("DEGRADED_HOLD", fixed_degraded_reason_);
    }
    else if (local_mode_ == "DEGRADED_HOLD")
    {
      if (degraded_bootstrap_enable_)
      {
        enterLocalReinitMode("DEGRADED_BOOTSTRAP", "exit_degraded_hold");
      }
      else
      {
        ROS_WARN("[LOCAL_REINIT] Exit DEGRADED_HOLD -> NORMAL: degraded_bootstrap disabled.");
        local_mode_ = "NORMAL";
        local_reinit_reason_ = "exit_degraded_hold_no_post_reinit";
        local_mode_start_time_ = now;
      }
    }
  }
  else if (local_mode_ == "DEGRADED_HOLD" || local_mode_ == "DEGRADED_BOOTSTRAP")
  {
    local_mode_ = "NORMAL";
    local_reinit_reason_ = "debug_fixed_degraded_disabled";
    local_mode_start_time_ = now;
  }

  if (local_mode_ == "DEGRADED_BOOTSTRAP" && !degraded_bootstrap_enable_)
  {
    local_mode_ = "NORMAL";
    local_reinit_reason_ = "degraded_bootstrap_disabled";
    local_mode_start_time_ = now;
  }

  if ((local_mode_ == "DEGRADED_BOOTSTRAP" || local_mode_ == "LOCAL_REINIT" || local_mode_ == "RELOCALIZE") &&
      local_mode_start_time_ >= 0.0)
  {
    const double mode_elapsed = now - local_mode_start_time_;
    const bool enough_time = local_post_reinit_duration_sec_ <= 0.0 || mode_elapsed >= local_post_reinit_duration_sec_;
    const bool enough_lio = lio_bootstrap_frames_ >= local_post_reinit_lio_frames_;
    const bool enough_vio = vio_bootstrap_frames_ >= local_post_reinit_vio_frames_ || !img_en;
    if (enough_time && enough_lio && enough_vio)
    {
      ROS_WARN("[LOCAL_REINIT] Exit %s -> NORMAL: elapsed=%.3f lio_bootstrap=%d vio_bootstrap=%d",
               local_mode_.c_str(), mode_elapsed, lio_bootstrap_frames_, vio_bootstrap_frames_);
      local_mode_ = "NORMAL";
      local_reinit_reason_ = "stable_bootstrap";
      local_mode_start_time_ = now;
      local_vio_low_start_time_ = -1.0;
      local_lio_weak_start_time_ = -1.0;
      local_vio_unavailable_ = false;
      local_lio_weak_ = false;
    }
  }
}

const char *LIVMapper::systemModeName(SystemMode mode)
{
  switch (mode)
  {
    case SystemMode::NORMAL: return "NORMAL";
    case SystemMode::DEGRADED_HOLD: return "DEGRADED_HOLD";
    case SystemMode::DEGRADED_BOOTSTRAP: return "DEGRADED_BOOTSTRAP";
    case SystemMode::LOCAL_REINIT: return "LOCAL_REINIT";
  }
  return "NORMAL";
}

const char *LIVMapper::rejectReasonName(RejectReason reason)
{
  switch (reason)
  {
    case RejectReason::NONE: return "NONE";
    case RejectReason::SAFETY: return "SAFETY";
    case RejectReason::DEGRADED_HOLD: return "DEGRADED_HOLD";
    case RejectReason::BACKWARD_SLIP: return "BACKWARD_SLIP";
    case RejectReason::LOCAL_REINIT: return "LOCAL_REINIT";
    case RejectReason::BOOTSTRAP: return "BOOTSTRAP";
    case RejectReason::SMALL_MOTION: return "SMALL_MOTION";
    case RejectReason::NO_POINTS: return "NO_POINTS";
    case RejectReason::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

LIVMapper::ObservationQuality LIVMapper::evaluateObservationQuality() const
{
  ObservationQuality quality;
  quality.speed = _state.vel_end.norm();
  const V3D rpy = RotMtoEuler(_state.rot_end);
  quality.roll_deg = rpy.x() * 57.29577951308232;
  quality.pitch_deg = rpy.y() * 57.29577951308232;
  quality.speed_abnormal = safety_max_speed_ > 0.0 && quality.speed > safety_max_speed_;
  quality.attitude_abnormal =
      degraded_hold_attitude_reject_deg_ > 0.0 &&
      (std::fabs(quality.roll_deg) > degraded_hold_attitude_reject_deg_ ||
       std::fabs(quality.pitch_deg) > degraded_hold_attitude_reject_deg_);

  if (voxelmap_manager)
  {
    quality.lio_degenerated = voxelmap_manager->isLidarDegenerated();
    quality.lio_valid =
        voxelmap_manager->last_update_status_ != "skipped" &&
        voxelmap_manager->last_update_status_ != "rejected";
    quality.map_update_weak =
        deg_guard_last_lio_voxel_map_skip_reason_ != "none" &&
        deg_guard_last_lio_voxel_map_skip_reason_ != "not_lio" &&
        deg_guard_last_lio_voxel_map_skip_reason_ != "forced_by_lidar_frames" &&
        deg_guard_last_lio_voxel_map_skip_reason_ != "forced_by_time";
  }
  if (vio_manager)
  {
    quality.vio_low_tracked = vio_manager->total_points < deg_guard_min_visual_tracked_points_;
    quality.vio_valid = !quality.vio_low_tracked;
  }
  if (uwb_manager)
  {
    const UwbUpdateSummary &uwb_summary = uwb_manager->lastUpdateSummary();
    quality.uwb_valid = !uwb_summary.relocalize_required;
    quality.uwb_residual_abnormal = uwb_summary.relocalize_required;
  }

  quality.backward_distance = std::max(corridor_prior_backward_distance_window_,
                                       safety_backward_distance_in_window_);
  quality.backward_slip =
      (corridor_prior_action_ == "block_map_update" ||
       corridor_prior_action_ == "local_reinit" ||
       corridor_prior_action_ == "downweight" ||
       quality.backward_distance > corridor_prior_backward_distance_threshold_);

  if (safety_fail_safe_mode_) quality.reason = "safety";
  else if (quality.backward_slip) quality.reason = "backward_slip";
  else if (quality.lio_degenerated) quality.reason = "lio_degenerated";
  else if (quality.vio_low_tracked) quality.reason = "vio_low_tracked";
  else if (quality.map_update_weak) quality.reason = "map_update_weak";
  else quality.reason = "normal";
  return quality;
}

void LIVMapper::updateSystemModeFromQuality(const ObservationQuality &quality)
{
  if (!experimental_features_enable_) return;
  if (!local_reinit_enable_) return;
  if (safety_fail_safe_mode_)
  {
    if (local_mode_ != "DEGRADED_HOLD") enterLocalReinitMode("DEGRADED_HOLD", "safety");
    return;
  }
  if (local_mode_ == "NORMAL" &&
      quality.backward_slip &&
      corridor_prior_action_ == "local_reinit")
  {
    enterLocalReinitMode("LOCAL_REINIT", "BACKWARD_LOCAL_REINIT");
  }
}

LIVMapper::UpdateDecision LIVMapper::makeUpdateDecision(const ObservationQuality &quality) const
{
  UpdateDecision decision;
  if (!experimental_features_enable_) return decision;
  if (local_mode_ == "DEGRADED_HOLD" || safety_fail_safe_mode_)
  {
    decision.allow_lio_update = false;
    decision.allow_vio_update = false;
    decision.allow_uwb_update = false;
    decision.allow_voxel_map_update = false;
    decision.allow_visual_map_update = false;
    decision.allow_publish = false;
    decision.mode = SystemMode::DEGRADED_HOLD;
    decision.reason = safety_fail_safe_mode_ ? RejectReason::SAFETY : RejectReason::DEGRADED_HOLD;
    decision.reason_text = safety_fail_safe_mode_ ? safety_last_reason_ : local_reinit_reason_;
    return decision;
  }

  if (local_mode_ == "DEGRADED_BOOTSTRAP")
  {
    decision.allow_vio_update = false;
    decision.mode = SystemMode::DEGRADED_BOOTSTRAP;
    decision.reason = RejectReason::BOOTSTRAP;
    decision.reason_text = local_reinit_reason_;
    return decision;
  }

  if (local_mode_ == "LOCAL_REINIT" || local_mode_ == "RELOCALIZE")
  {
    decision.allow_vio_update = false;
    decision.allow_uwb_update = local_mode_ != "RELOCALIZE";
    decision.mode = SystemMode::LOCAL_REINIT;
    decision.reason = RejectReason::LOCAL_REINIT;
    decision.reason_text = local_reinit_reason_;
    return decision;
  }

  decision.mode = SystemMode::NORMAL;
  if (quality.backward_slip &&
      (corridor_prior_action_ == "block_map_update" || corridor_prior_action_ == "downweight"))
  {
    decision.allow_voxel_map_update = corridor_prior_update_voxel_map_enabled_;
    decision.allow_visual_map_update = corridor_prior_visual_map_update_enabled_;
    decision.reason = RejectReason::BACKWARD_SLIP;
    decision.reason_text = "BACKWARD_SLIP";
  }
  return decision;
}

LIVMapper::MapUpdateDecision LIVMapper::decideMapUpdate(bool lio_degenerated,
                                                        bool stride_ready,
                                                        bool force_ready,
                                                        bool has_points) const
{
  MapUpdateDecision decision;
  if (!experimental_features_enable_)
  {
    decision.allow = has_points && (stride_ready || force_ready);
    if (!has_points)
    {
      decision.reason = RejectReason::NO_POINTS;
      decision.skip_reason = "NO_POINTS";
    }
    else if (!decision.allow)
    {
      decision.reason = RejectReason::SMALL_MOTION;
      decision.skip_reason = "SMALL_MOTION";
    }
    else
    {
      decision.reason = RejectReason::NONE;
      decision.skip_reason = force_ready && !stride_ready ? "forced" : "none";
    }
    return decision;
  }
  decision.allow = current_decision_.allow_voxel_map_update && !lio_degenerated && stride_ready && has_points;
  if (!current_decision_.allow_voxel_map_update)
  {
    decision.reason = current_decision_.reason;
    decision.skip_reason = current_decision_.reason == RejectReason::BACKWARD_SLIP ? "BACKWARD_SLIP" :
                           current_decision_.reason == RejectReason::DEGRADED_HOLD ? "DEGRADED_HOLD" :
                           current_decision_.reason == RejectReason::BOOTSTRAP ? "BOOTSTRAP" :
                           current_decision_.reason == RejectReason::LOCAL_REINIT ? "LOCAL_REINIT" :
                           current_decision_.reason == RejectReason::SAFETY ? "DEGRADED_HOLD" : "UNKNOWN";
    return decision;
  }
  if (!has_points)
  {
    decision.reason = RejectReason::NO_POINTS;
    decision.skip_reason = "NO_POINTS";
    return decision;
  }
  if (lio_degenerated)
  {
    decision.reason = RejectReason::BACKWARD_SLIP;
    decision.skip_reason = current_quality_.backward_slip ? "BACKWARD_SLIP" : "UNKNOWN";
    return decision;
  }
  if (!stride_ready && !force_ready)
  {
    decision.reason = RejectReason::SMALL_MOTION;
    decision.skip_reason = "SMALL_MOTION";
    return decision;
  }
  if (!stride_ready && force_ready)
  {
    decision.allow = true;
    decision.reason = RejectReason::NONE;
    decision.skip_reason = "forced";
    return decision;
  }
  decision.allow = true;
  decision.reason = RejectReason::NONE;
  decision.skip_reason = "none";
  return decision;
}

void LIVMapper::printDiagnostics(const char *stage)
{
  if (diagnostics_level_ == "off") return;
  const double now = LidarMeasures.last_lio_update_time;
  if (diagnostics_level_ == "summary")
  {
    if (diagnostics_last_summary_time_ >= 0.0 &&
        now - diagnostics_last_summary_time_ < diagnostics_summary_interval_sec_)
    {
      return;
    }
    diagnostics_last_summary_time_ = now;
    ROS_INFO("[Diagnostics] mode=%s lio_valid=%d vio_valid=%d uwb_valid=%d map_update=%d backward_distance=%.3f speed=%.3f roll=%.2f pitch=%.2f decision_reason=%s stage=%s",
             systemModeName(current_decision_.mode),
             static_cast<int>(current_quality_.lio_valid),
             static_cast<int>(current_quality_.vio_valid),
             static_cast<int>(current_quality_.uwb_valid),
             static_cast<int>(current_decision_.allow_voxel_map_update),
             current_quality_.backward_distance,
             current_quality_.speed,
             current_quality_.roll_deg,
             current_quality_.pitch_deg,
             current_decision_.reason_text.c_str(),
             stage ? stage : "frame");
  }
}

void LIVMapper::updateLocalTrackingLostDetectors(const char *sensor)
{
  if (!experimental_features_enable_) return;
  if (!local_reinit_enable_) return;
  const double now = LidarMeasures.last_lio_update_time;
  const std::string sensor_name = sensor ? sensor : "";
  if (sensor_name == "VIO")
  {
    const bool low = deg_guard_last_visual_tracked_points_ >= 0 &&
                     deg_guard_last_visual_tracked_points_ < local_vio_unavailable_tracked_points_;
    if (low)
    {
      if (local_vio_low_start_time_ < 0.0) local_vio_low_start_time_ = now;
      local_vio_unavailable_ = now - local_vio_low_start_time_ >= local_tracking_lost_window_sec_;
    }
    else
    {
      local_vio_low_start_time_ = -1.0;
      local_vio_unavailable_ = false;
    }
  }
  else if (sensor_name == "LIO")
  {
    const bool weak = !deg_guard_last_lio_voxel_map_updated_ ||
                      (local_lio_weak_residual_threshold_ > 0.0 &&
                       voxelmap_manager &&
                       voxelmap_manager->last_average_residual_ > local_lio_weak_residual_threshold_);
    if (weak)
    {
      if (local_lio_weak_start_time_ < 0.0) local_lio_weak_start_time_ = now;
      local_lio_weak_ = now - local_lio_weak_start_time_ >= local_tracking_lost_window_sec_;
    }
    else
    {
      local_lio_weak_start_time_ = -1.0;
      local_lio_weak_ = false;
    }
  }

  if (local_mode_ == "NORMAL" && local_vio_unavailable_ && local_lio_weak_)
  {
    enterLocalReinitMode("LOCAL_REINIT", "local_tracking_lost");
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();
  const StatesGroup state_before_imu = _state;
  skip_mapping_this_frame_ = false;
  if (safety_guard_enable_ && safety_fail_safe_mode_)
  {
    if (safety_reliable_state_ready_) _state = safety_reliable_state_;
    _state.vel_end.setZero();
    state_propagat = _state;
    if (voxelmap_manager) voxelmap_manager->state_ = _state;
    if (vio_manager) vio_manager->updateFrameState(_state);
    return;
  }
  if (!validateStateForSafety("IMU-pre", state_before_imu, _state, false, false)) return;

  const bool imu_was_initializing = p_imu && p_imu->imu_need_init;
  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (!validateStateForSafety("IMU-post", state_before_imu, _state, false, false))
  {
    return;
  }

  if (gravity_align_en) gravityAlignment();

  if (!validateStateForSafety("gravity-post", state_before_imu, _state, false, false))
  {
    return;
  }

  snapStateForDeterminism(_state);
  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;
  if (safety_guard_enable_ && imu_was_initializing)
  {
    // ponytail: the IMU init/gravity-align frame has no undistorted scan yet; start mapping on the next synced frame.
    skip_mapping_this_frame_ = true;
  }

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  static int vio_dispatch_count = 0;
  static int lio_dispatch_count = 0;

  deterministic_frame_id_++;
  updateLocalModeAtFrameStart();
  const StatesGroup state_before_frame = _state;
  if (experimental_features_enable_)
  {
    validateStateForSafety("frame-pre", state_before_frame, _state, false, false);
    updateCorridorMotionPrior("frame-pre", _state);
    current_quality_ = evaluateObservationQuality();
    updateSystemModeFromQuality(current_quality_);
    current_decision_ = makeUpdateDecision(current_quality_);
    update_decision_ready_ = true;
    printDiagnostics("frame-pre");
  }
  else
  {
    current_quality_ = ObservationQuality();
    current_decision_ = UpdateDecision();
    update_decision_ready_ = false;
  }

  switch (LidarMeasures.lio_vio_flg)
  {
    case VIO:
      vio_dispatch_count++;
      if (vio_dispatch_count % 20 == 1)
      {
        std::cout << "[ Flow ] Dispatch VIO frame #" << vio_dispatch_count
                  << " (LIO dispatched=" << lio_dispatch_count << ")" << std::endl;
      }
      handleVIO();
      break;
    case LIO:
    case LO:
      lio_dispatch_count++;
      if (lio_dispatch_count % 50 == 1)
      {
        std::cout << "[ Flow ] Dispatch LIO/LO frame #" << lio_dispatch_count
                  << " (VIO dispatched=" << vio_dispatch_count << ")" << std::endl;
      }
      handleLIO();
      break;
  }
  if (experimental_features_enable_)
  {
    maybeRecoverFailSafe();
    current_quality_ = evaluateObservationQuality();
    current_decision_ = makeUpdateDecision(current_quality_);
    writeDegeneracyGuardLog("frame-end");
    printDiagnostics("frame-end");
  }
  snapStateForDeterminism(_state);
  voxelmap_manager->state_ = _state;
  if (state_update_flg) latest_ekf_state = _state;
}

void LIVMapper::applyUwbUpdate(const char *stage)
{
  if (!uwb_manager || !uwb_manager->updateEnabled()) return;
  const bool is_lio_frame = LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO;
  if (uwb_update_only_on_lio_ && !is_lio_frame) return;
  if (update_decision_ready_ && !current_decision_.allow_uwb_update) return;
  if (safety_guard_enable_ && safety_fail_safe_mode_) return;
  if (uwb_skip_when_lio_frozen_ && lio_freeze_state_ready_)
  {
    ROS_WARN_THROTTLE(1.0, "[UWB] Skip update while LIO state is frozen.");
    return;
  }

  uwb_output_target_offset_.z() = 0.0;
  uwb_output_pos_offset_.z() = 0.0;
  const V3D pos_before = _state.pos_end;
  const int used_count =
      uwb_manager->applyRangeUpdateAt(_state, LidarMeasures.last_lio_update_time, uwb_update_window_sec_);
  const UwbUpdateSummary &summary = uwb_manager->lastUpdateSummary();
  std::ostringstream anchor_ids;
  for (size_t i = 0; i < summary.used_anchor_ids.size(); ++i)
  {
    if (i > 0) anchor_ids << "|";
    anchor_ids << summary.used_anchor_ids[i];
  }
  deterministic_last_uwb_anchor_ids_ = anchor_ids.str();
  deterministic_last_uwb_update_ = used_count > 0;

  if (summary.relocalize_required ||
      (uwb_relocalize_en_ &&
       uwb_relocalize_xy_threshold_ > 0.0 &&
       summary.xy_correction_before_step > uwb_relocalize_xy_threshold_))
  {
    enterLocalReinitMode("RELOCALIZE", "uwb_xy_correction_over_threshold");
    return;
  }
  if (used_count <= 0) return;

  V3D output_delta = _state.pos_end - pos_before;
  output_delta.z() = 0.0;
  uwb_output_target_offset_.setZero();
  uwb_output_pos_offset_.setZero();
  applyDegeneracyGuardCorrections("UWB-XY");

  snapStateForDeterminism(_state);
  voxelmap_manager->state_ = _state;
  if (vio_manager) vio_manager->updateFrameState(_state);

  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  ROS_INFO_THROTTLE(1.0,
                    "[UWB] Applied XY update used=%d after %s stamp=%.6f anchors=%s xy_delta=%.4f z_before=%.4f z_after=%.4f.",
                    used_count,
                    stage ? stage : "state",
                    LidarMeasures.last_lio_update_time,
                    deterministic_last_uwb_anchor_ids_.c_str(),
                    std::hypot(output_delta.x(), output_delta.y()),
                    summary.z_correction_before_clamp,
                    summary.z_correction_after_clamp);
  if (vio_manager)
  {
    std::vector<std::string> lines;
    std::ostringstream oss;
    oss << "[ UWB ] Applied XY update used=" << used_count
        << " after " << (stage ? stage : "state")
        << " stamp=" << std::fixed << std::setprecision(6) << LidarMeasures.last_lio_update_time
        << " anchors=" << deterministic_last_uwb_anchor_ids_
        << " xy_delta=" << std::hypot(output_delta.x(), output_delta.y())
        << " z_before=" << summary.z_correction_before_clamp
        << " z_after=" << summary.z_correction_after_clamp;
    lines.push_back(oss.str());
    vio_manager->appendTimingLogLines(lines);
  }
}

void LIVMapper::advanceUwbOutputCorrection()
{
  if (!uwb_output_correction_en_) return;
  uwb_output_target_offset_.z() = 0.0;
  uwb_output_pos_offset_.z() = 0.0;
  if (!uwb_output_smooth_en_)
  {
    uwb_output_pos_offset_ = uwb_output_target_offset_;
    uwb_output_pos_offset_.z() = 0.0;
    return;
  }

  const V3D residual = uwb_output_target_offset_ - uwb_output_pos_offset_;
  if (residual.norm() < 1e-6) return;

  V3D step = residual * uwb_output_smooth_alpha_;
  const double step_norm = step.norm();
  if (uwb_output_smooth_max_step_m_ > 0.0 && step_norm > uwb_output_smooth_max_step_m_)
  {
    step *= uwb_output_smooth_max_step_m_ / std::max(step_norm, 1e-9);
  }
  uwb_output_pos_offset_ += step;
  uwb_output_pos_offset_.z() = 0.0;
}

V3D LIVMapper::outputPosition() const
{
  V3D out_pos = uwb_output_correction_en_ ? (_state.pos_end + uwb_output_pos_offset_) : _state.pos_end;
  out_pos.z() = _state.pos_end.z();
  return out_pos;
}

void LIVMapper::handleVIO() 
{
  if (safety_guard_enable_ && safety_fail_safe_mode_)
  {
    deg_guard_last_sensor_type_ = "VIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ = "safety_fail_safe";
    deg_guard_last_action_ = "fail_safe";
    deg_guard_last_reason_ = safety_last_reason_;
    deg_guard_last_lio_voxel_map_updated_ = false;
    deg_guard_last_lio_voxel_map_skip_reason_ = "DEGRADED_HOLD";
    clearSafetyLocalCaches();
    return;
  }
	  if (vio_manager == nullptr || LidarMeasures.measures.empty())
	  {
    deg_guard_last_sensor_type_ = "VIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ = (vio_manager == nullptr) ? "null_vio_manager" : "empty_measure_group";
    deg_guard_last_action_ = "none";
    deg_guard_last_reason_ = deg_guard_last_reject_reason_;
	    return;
	  }

  if (isDegradedHoldMode())
  {
    const StatesGroup state_before_degraded_hold = _state;
    applyDegradedHoldConstraint("VIO-DEGRADED_HOLD", true);
    deg_guard_last_sensor_type_ = "VIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ =
        degraded_hold_last_reject_reason_ == "none" ? "DEGRADED_HOLD" : degraded_hold_last_reject_reason_;
    deg_guard_last_action_ = "degraded_hold";
    deg_guard_last_reason_ = "DEGRADED_HOLD";
    deg_guard_last_visual_tracked_points_ = -1;
    deg_guard_last_final_pose_delta_ = (_state.pos_end - state_before_degraded_hold.pos_end).norm();
    deterministic_last_vio_update_ = false;
    if (vio_manager)
    {
      vio_manager->disable_visual_map_update_this_frame = true;
      vio_manager->visual_map_update_disable_reason = "DEGRADED_HOLD";
      vio_manager->force_skip_visual_ekf_this_frame = true;
      vio_manager->force_skip_visual_ekf_reason = "DEGRADED_HOLD";
      vio_manager->last_visual_update_status = "skipped";
      vio_manager->last_visual_update_reject_reason = deg_guard_last_reject_reason_;
    }
    if (imu_prop_enable)
    {
      ekf_finish_once = true;
      latest_ekf_state = _state;
      latest_ekf_time = LidarMeasures.last_lio_update_time;
      state_update_flg = true;
    }
    ROS_WARN_THROTTLE(0.5,
                      "[LOCAL_REINIT] DEGRADED_HOLD skip VIO pose/map update: local_elapsed=%.3f bag_elapsed=%.3f reason=%s",
                      local_elapsed_sec_, bag_elapsed_sec_, deg_guard_last_reject_reason_.c_str());
    return;
  }

  if (update_decision_ready_ &&
      !current_decision_.allow_vio_update &&
      !current_decision_.allow_visual_map_update)
  {
    deg_guard_last_sensor_type_ = "VIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ = current_decision_.reason_text;
    deg_guard_last_action_ = "decision_skip";
    deg_guard_last_reason_ = current_decision_.reason_text;
    deterministic_last_vio_update_ = false;
    return;
  }

	  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if ((pcl_w_wait_pub == nullptr) || pcl_w_wait_pub->empty())
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  const bool use_visual_frame = shouldSelectVisualFrame();
	  if (!use_visual_frame)
	  {
	    deterministic_last_vio_update_ = false;
	    if (deg_guard_enable_)
    {
      deg_guard_last_sensor_type_ = "VIO";
      deg_guard_last_update_status_ = "skipped";
      deg_guard_last_reject_reason_ = "visual_selector:" + last_selector_reason_;
      deg_guard_last_action_ = "none";
      deg_guard_last_reason_ = deg_guard_last_reject_reason_;
      deg_guard_last_visual_tracked_points_ = -1;
      deg_guard_last_update_translation_norm_ = 0.0;
      deg_guard_last_update_translation_rate_mps_ = 0.0;
      deg_guard_last_update_yaw_deg_ = 0.0;
      deg_guard_last_update_yaw_rate_degps_ = 0.0;
      deg_guard_last_visual_update_rot_deg_ = 0.0;
      deg_guard_last_visual_update_rot_rate_degps_ = 0.0;
      deg_guard_last_final_pose_delta_ = 0.0;
      const double current_time = LidarMeasures.last_lio_update_time;
      deg_guard_last_dt_ =
          (deg_guard_last_vio_time_ >= 0.0 && current_time > deg_guard_last_vio_time_) ?
          (current_time - deg_guard_last_vio_time_) : deg_guard_camera_dt_;
      deg_guard_last_dt_ = std::max(1e-4, deg_guard_last_dt_);
      deg_guard_last_vio_time_ = current_time;
      deg_guard_last_vio_skip_affects_degenerate_ = false;
    }

    static int adaptive_skip_count = 0;
    adaptive_skip_count++;
    if (adaptive_skip_count % 10 == 1)
    {
      std::cout << "[ VIO ] Skip by selector: reason=" << last_selector_reason_
                << ", count=" << adaptive_skip_count
                << ", delta(t/r)=" << last_selector_trans_delta_ << "m/" << last_selector_rot_delta_deg_ << "deg"
                << ", thresh(t/r)=" << last_selector_trans_thresh_ << "m/" << last_selector_rot_thresh_deg_ << "deg"
                << ", ratio=" << last_selector_constraint_ratio_
                << ", skip=" << skipped_visual_frames_ << "/" << keyframe_max_skip_frames_
                << std::endl;
    }

    if (imu_prop_enable)
    {
      ekf_finish_once = true;
      latest_ekf_state = _state;
      latest_ekf_time = LidarMeasures.last_lio_update_time;
      state_update_flg = true;
    }

    if (vio_manager)
    {
      vio_manager->last_visual_guard_time = LidarMeasures.last_lio_update_time - _first_lidar_time;
      vio_manager->last_visual_guard_pos = _state.pos_end;
      vio_manager->has_last_visual_guard_pos = true;
    }

	    if (!deterministic_mode_ || !uwb_update_only_on_lio_) applyUwbUpdate("VIO-skip");
	    advanceUwbOutputCorrection();
    applyDegeneracyGuardCorrections("VIO-skip");

    if ((!update_decision_ready_ || current_decision_.allow_publish) &&
        !(safety_guard_enable_ && safety_fail_safe_mode_))
    {
      publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
    }

    euler_cur = RotMtoEuler(_state.rot_end);
    const V3D out_pos = outputPosition();
    fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
             << out_pos.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
             << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
             << (feats_undistort ? feats_undistort->points.size() : 0) << std::endl;

    if (vio_manager)
    {
      auto formatDouble6 = [](double value)
      {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value;
        return oss.str();
      };

      auto makeTableRow = [](const std::string &left, const std::string &right)
      {
        std::ostringstream oss;
        oss << "| " << std::left << std::setw(29) << left
            << " | " << std::left << std::setw(27) << right << " |";
        return oss.str();
      };

      std::vector<std::string> lines;
      std::ostringstream oss;
      oss << "[ VIO ] Skip by selector: reason=" << last_selector_reason_
          << ", count=" << adaptive_skip_count
          << ", delta(t/r)=" << formatDouble6(last_selector_trans_delta_) << "m/" << formatDouble6(last_selector_rot_delta_deg_) << "deg"
          << ", thresh(t/r)=" << formatDouble6(last_selector_trans_thresh_) << "m/" << formatDouble6(last_selector_rot_thresh_deg_) << "deg"
          << ", ratio=" << formatDouble6(last_selector_constraint_ratio_)
          << ", skip=" << skipped_visual_frames_ << "/" << keyframe_max_skip_frames_;
      lines.push_back(oss.str());
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Raw Feature Num", std::to_string(pcl_w_wait_pub->points.size())));
      lines.push_back(makeTableRow("Selector Reason", last_selector_reason_));
      lines.push_back(makeTableRow("Skip Count", std::to_string(adaptive_skip_count)));
      lines.push_back(makeTableRow("Skip Limit", std::to_string(keyframe_max_skip_frames_)));
      lines.push_back(makeTableRow("Delta t/r", formatDouble6(last_selector_trans_delta_) + "m/" + formatDouble6(last_selector_rot_delta_deg_) + "deg"));
      lines.push_back(makeTableRow("Thresh t/r", formatDouble6(last_selector_trans_thresh_) + "m/" + formatDouble6(last_selector_rot_thresh_deg_) + "deg"));
      lines.push_back(makeTableRow("Constraint Ratio", formatDouble6(last_selector_constraint_ratio_)));
      lines.push_back("+-------------------------------------------------------------+");
      vio_manager->appendTimingLogLines(lines);
    }
    return;
  }

  if (lio_freeze_state_when_degenerate_ && lio_freeze_state_ready_)
  {
    // ponytail: while LIO is frozen, VIO can only fight the freeze; skip it until LIO is observable again.
    _state = lio_freeze_state_;
    _state.vel_end.setZero();
    applyDegeneracyGuardCorrections("VIO-freeze");
    snapStateForDeterminism(_state);
    voxelmap_manager->state_ = _state;
    if (vio_manager)
    {
      vio_manager->updateFrameState(_state);
      vio_manager->last_visual_guard_time = LidarMeasures.last_lio_update_time - _first_lidar_time;
      vio_manager->last_visual_guard_pos = _state.pos_end;
      vio_manager->has_last_visual_guard_pos = true;
    }

    ROS_WARN_THROTTLE(1.0, "[VIO] Skip visual update while LIO state is frozen by degenerated constraints.");

	    if (!deterministic_mode_ || !uwb_update_only_on_lio_) applyUwbUpdate("VIO-freeze");
	    advanceUwbOutputCorrection();

    if (imu_prop_enable)
    {
      ekf_finish_once = true;
      latest_ekf_state = _state;
      latest_ekf_time = LidarMeasures.last_lio_update_time;
      state_update_flg = true;
    }

    if ((!update_decision_ready_ || current_decision_.allow_publish) &&
        !(safety_guard_enable_ && safety_fail_safe_mode_))
    {
      publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
    }

    euler_cur = RotMtoEuler(_state.rot_end);
    const V3D out_pos = outputPosition();
    fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
             << out_pos.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
             << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
             << (feats_undistort ? feats_undistort->points.size() : 0) << std::endl;
    return;
  }

		  const StatesGroup state_before_vio_update = _state;
		  if (vio_manager)
		  {
		    auto visualModeReason = [&]() {
		      if (local_mode_ == "DEGRADED_HOLD") return std::string("DEGRADED_HOLD");
		      if (local_mode_ == "DEGRADED_BOOTSTRAP" ||
		          local_mode_ == "LOCAL_REINIT" ||
		          local_mode_ == "RELOCALIZE")
		      {
		        return std::string("BOOTSTRAP");
		      }
		      return std::string("");
		    };
		    vio_manager->disable_visual_map_update_this_frame =
		        experimental_features_enable_ &&
		        ((update_decision_ready_ && !current_decision_.allow_visual_map_update) ||
		         localModeBlocksVisualMapUpdate());
		    vio_manager->visual_map_update_disable_reason =
		        vio_manager->disable_visual_map_update_this_frame ?
		        (localModeBlocksVisualMapUpdate() ? visualModeReason() : current_decision_.reason_text) : "";
		    vio_manager->force_skip_visual_ekf_this_frame =
		        experimental_features_enable_ &&
		        (localModeSkipsVisualEkf() ||
		         (update_decision_ready_ && !current_decision_.allow_vio_update));
		    vio_manager->force_skip_visual_ekf_reason =
		        vio_manager->force_skip_visual_ekf_this_frame ?
		        (!current_decision_.allow_vio_update ? current_decision_.reason_text : visualModeReason()) : "";
		    vio_manager->adaptive_external_noise_scale =
		        experimental_features_enable_ ? updateAdaptiveSensorNoiseScale("VIO") : 1.0;
		  }
	  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);
	  const std::string vio_update_status = vio_manager ? vio_manager->last_visual_update_status : "accepted";
  const std::string vio_reject_reason = vio_manager ? vio_manager->last_visual_update_reject_reason : "";
  const std::string vio_guard_reason = vio_manager ? vio_manager->last_visual_update_guard_reason : "";
  if (vio_manager)
  {
    deg_guard_last_vio_noise_scale_ = vio_manager->last_adaptive_img_point_cov_scale;
    deg_guard_last_weight_reason_ = vio_manager->last_adaptive_weight_reason;
	  }
	  evaluateDegeneracyGuardUpdate("VIO", state_before_vio_update, -1, vio_manager ? vio_manager->total_points : -1);
	  deterministic_last_vio_update_ = !(vio_update_status == "skipped" || vio_update_status == "rejected");
	  if (local_mode_ == "DEGRADED_BOOTSTRAP" || local_mode_ == "LOCAL_REINIT" || local_mode_ == "RELOCALIZE")
	  {
	    vio_bootstrap_frames_++;
	  }
	  updateLocalTrackingLostDetectors("VIO");
  if ((vio_update_status == "skipped" || vio_update_status == "rejected" || vio_update_status == "downweighted") &&
      deg_guard_last_update_status_ == "accepted")
  {
    deg_guard_last_update_status_ = vio_update_status;
    deg_guard_last_reject_reason_ = !vio_reject_reason.empty() ? vio_reject_reason : vio_guard_reason;
  }
  else if (!vio_guard_reason.empty() && deg_guard_last_reject_reason_.empty())
  {
    deg_guard_last_reject_reason_ = vio_guard_reason;
  }
  applyDegeneracyGuardCorrections("VIO");
  snapStateForDeterminism(_state);
  if (!validateStateForSafety("VIO-post", state_before_vio_update, _state, true, false))
  {
    if (vio_manager) vio_manager->updateFrameState(_state);
    return;
  }
  vio_manager->updateFrameState(_state);
  if (experimental_features_enable_) updateVisualObservationHints();
	  if (!deterministic_mode_ || !uwb_update_only_on_lio_) applyUwbUpdate("VIO");
	  advanceUwbOutputCorrection();
  if (!validateStateForSafety("VIO-after-uwb", state_before_vio_update, _state, false, false))
  {
    if (vio_manager) vio_manager->updateFrameState(_state);
    return;
  }

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  if ((!update_decision_ready_ || current_decision_.allow_publish) &&
      validateStateForSafety("VIO-pre-publish", state_before_vio_update, _state, false, false))
  {
    publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
  }
  if (!suppress_image_pub_)
  {
    publish_img_counter_++;
    if (publish_img_counter_ % std::max(1, publish_img_stride_) == 0)
    {
      publish_img_rgb(pubImage, vio_manager);
    }
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  const V3D out_pos = outputPosition();
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << out_pos.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
            << (feats_undistort ? feats_undistort->points.size() : 0) << std::endl;

}

bool LIVMapper::shouldSelectVisualFrame()
{
  last_selector_constraint_ratio_ = 0.0;
  last_selector_trans_thresh_ = 0.0;
  last_selector_rot_thresh_deg_ = 0.0;
  last_selector_trans_delta_ = 0.0;
  last_selector_rot_delta_deg_ = 0.0;
  last_selector_reach_pose_keyframe_ = false;
  last_selector_reach_skip_limit_ = false;

  if (!experimental_features_enable_ || !adaptive_visual_selector_en)
  {
    last_selector_reason_ = "adaptive_off";
    return true;
  }
  if (!img_en)
  {
    last_selector_reason_ = "img_disabled";
    return false;
  }

  if (voxelmap_manager->isLidarDegenerated())
  {
    last_selector_reason_ = "lidar_degenerated_force";
    skipped_visual_frames_ = 0;
    has_last_visual_keyframe_state_ = true;
    last_visual_keyframe_state_ = _state;
    return true;
  }

  if (!has_last_visual_keyframe_state_)
  {
    last_selector_reason_ = "first_visual_keyframe";
    has_last_visual_keyframe_state_ = true;
    last_visual_keyframe_state_ = _state;
    skipped_visual_frames_ = 0;
    return true;
  }

  const double ratio_full = std::max(1e-6, keyframe_constraint_ratio_full_);
  const double constraint_ratio = std::max(0.0, std::min(1.0, voxelmap_manager->getLidarConstraintRatio() / ratio_full));
  const double trans_thresh = keyframe_trans_thresh_min_ +
                              constraint_ratio * (keyframe_trans_thresh_max_ - keyframe_trans_thresh_min_);
  const double rot_thresh_deg = keyframe_rot_thresh_min_deg_ +
                                constraint_ratio * (keyframe_rot_thresh_max_deg_ - keyframe_rot_thresh_min_deg_);

  const double trans_delta = (_state.pos_end - last_visual_keyframe_state_.pos_end).norm();
  Eigen::Matrix3d dR = last_visual_keyframe_state_.rot_end.transpose() * _state.rot_end;
  const double rot_delta_deg = Eigen::AngleAxisd(dR).angle() * 57.29577951308232;

  const bool reach_pose_keyframe = (trans_delta >= trans_thresh) || (rot_delta_deg >= rot_thresh_deg);
  const bool reach_skip_limit = skipped_visual_frames_ >= keyframe_max_skip_frames_;

  last_selector_constraint_ratio_ = constraint_ratio;
  last_selector_trans_thresh_ = trans_thresh;
  last_selector_rot_thresh_deg_ = rot_thresh_deg;
  last_selector_trans_delta_ = trans_delta;
  last_selector_rot_delta_deg_ = rot_delta_deg;
  last_selector_reach_pose_keyframe_ = reach_pose_keyframe;
  last_selector_reach_skip_limit_ = reach_skip_limit;

  if (reach_pose_keyframe || reach_skip_limit)
  {
    last_selector_reason_ = reach_pose_keyframe ? "pose_keyframe" : "max_skip";
    last_visual_keyframe_state_ = _state;
    skipped_visual_frames_ = 0;
    return true;
  }

  last_selector_reason_ = "below_threshold";
  skipped_visual_frames_++;
  return false;
}

void LIVMapper::updateVisualObservationHints()
{
  std::vector<VOXEL_LOCATION> observed_voxels;
  observed_voxels.reserve(vio_manager->feat_map.size());
  for (const auto &kv : vio_manager->feat_map)
  {
    if (kv.second != nullptr && !kv.second->voxel_points.empty())
    {
      observed_voxels.push_back(kv.first);
    }
  }
  if (deterministic_visual_observed_voxel_sort_en_)
  {
    std::sort(observed_voxels.begin(), observed_voxels.end(),
              [](const VOXEL_LOCATION &a, const VOXEL_LOCATION &b) {
                if (a.x != b.x) return a.x < b.x;
                if (a.y != b.y) return a.y < b.y;
                return a.z < b.z;
              });
  }
  voxelmap_manager->setVisualObservedVoxels(observed_voxels);
}

void LIVMapper::handleLIO() 
{    
  if (safety_guard_enable_ && safety_fail_safe_mode_)
  {
    deg_guard_last_sensor_type_ = "LIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ = "safety_fail_safe";
    deg_guard_last_action_ = "fail_safe";
    deg_guard_last_reason_ = safety_last_reason_;
    deg_guard_last_lio_update_executed_ = false;
    deg_guard_last_lio_voxel_map_updated_ = false;
    deg_guard_last_lio_voxel_map_skip_reason_ = "DEGRADED_HOLD";
    clearSafetyLocalCaches();
    return;
  }
  if (!validateStateForSafety("LIO-pre", _state, _state, false, false)) return;

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
	  if ((feats_undistort == nullptr) || feats_undistort->empty())
	  {
	    std::cout << "[ LIO ]: No point!!!" << std::endl;
	    return;
	  }

  if (isDegradedHoldMode())
  {
    const StatesGroup state_before_degraded_hold = _state;
    applyDegradedHoldConstraint("LIO-DEGRADED_HOLD", true);
    deg_guard_last_sensor_type_ = "LIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ =
        degraded_hold_last_reject_reason_ == "none" ? "DEGRADED_HOLD" : degraded_hold_last_reject_reason_;
    deg_guard_last_action_ = "degraded_hold";
    deg_guard_last_reason_ = "DEGRADED_HOLD";
    deg_guard_last_lio_update_executed_ = false;
    deterministic_last_lio_update_ = false;
    deg_guard_last_lio_voxel_map_updated_ = false;
    deg_guard_last_lio_voxel_map_skip_reason_ = "DEGRADED_HOLD";
    deg_guard_last_final_pose_delta_ = (_state.pos_end - state_before_degraded_hold.pos_end).norm();
    if (imu_prop_enable)
    {
      ekf_finish_once = true;
      latest_ekf_state = _state;
      latest_ekf_time = LidarMeasures.last_lio_update_time;
      state_update_flg = true;
    }
    euler_cur = RotMtoEuler(_state.rot_end);
    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
    if (!update_decision_ready_ || current_decision_.allow_publish) publish_odometry(pubOdomAftMapped);
    ROS_WARN_THROTTLE(0.5,
                      "[LOCAL_REINIT] DEGRADED_HOLD skip LIO pose/map update: local_elapsed=%.3f bag_elapsed=%.3f reason=%s",
                      local_elapsed_sec_, bag_elapsed_sec_, deg_guard_last_reject_reason_.c_str());
    return;
  }

  if (update_decision_ready_ && !current_decision_.allow_lio_update)
  {
    deg_guard_last_sensor_type_ = "LIO";
    deg_guard_last_update_status_ = "skipped";
    deg_guard_last_reject_reason_ = current_decision_.reason_text;
    deg_guard_last_action_ = "decision_skip";
    deg_guard_last_reason_ = current_decision_.reason_text;
    deg_guard_last_lio_update_executed_ = false;
    deterministic_last_lio_update_ = false;
    deg_guard_last_lio_voxel_map_updated_ = false;
    deg_guard_last_lio_voxel_map_skip_reason_ = rejectReasonName(current_decision_.reason);
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  if (deterministic_lio_feature_sort_en_)
  {
    std::sort(feats_down_body->points.begin(), feats_down_body->points.end(),
              [](const PointType &a, const PointType &b) {
                if (a.x != b.x) return a.x < b.x;
                if (a.y != b.y) return a.y < b.y;
                return a.z < b.z;
              });
  }
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  const StatesGroup state_before_lio_update = _state;
  voxelmap_manager->adaptive_external_noise_scale_ = updateAdaptiveSensorNoiseScale("LIO");
  voxelmap_manager->StateEstimation(state_propagat);
  deg_guard_last_lio_noise_scale_ = voxelmap_manager->last_adaptive_noise_scale_;
  deg_guard_last_weight_reason_ = voxelmap_manager->last_adaptive_weight_reason_;
	  const std::string lio_update_status = voxelmap_manager->last_update_status_;
	  const std::string lio_reject_reason = voxelmap_manager->last_reject_reason_;
	  deg_guard_last_lio_update_executed_ = !(lio_update_status == "skipped" || lio_update_status == "rejected");
	  deterministic_last_lio_update_ = deg_guard_last_lio_update_executed_;
	  deg_guard_last_lio_downweighted_ = deg_guard_last_lio_noise_scale_ > 1.001;
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;
  evaluateDegeneracyGuardUpdate("LIO", state_before_lio_update, voxelmap_manager->effct_feat_num_, -1);
  deg_guard_last_lio_downweighted_ =
      deg_guard_last_lio_downweighted_ || (deg_guard_last_update_status_ == "downweighted");
  if ((lio_update_status == "skipped" || lio_update_status == "rejected") &&
      deg_guard_last_update_status_ == "accepted")
  {
    deg_guard_last_update_status_ = lio_update_status;
    deg_guard_last_reject_reason_ = lio_reject_reason;
  }
  applyDegeneracyGuardCorrections("LIO");
  snapStateForDeterminism(_state);
  voxelmap_manager->state_ = _state;
  if (!validateStateForSafety("LIO-post", state_before_lio_update, _state, true, false))
  {
    if (vio_manager) vio_manager->updateFrameState(_state);
    return;
  }
  const bool lio_pose_finite = _state.pos_end.allFinite() && _state.rot_end.allFinite();
  double lio_state_jump_trans_m = 0.0;
  double lio_state_jump_rot_deg = 0.0;
  bool lio_state_jump = false;
  if (lio_state_jump_guard_en_)
  {
    if (!lio_pose_finite)
    {
      lio_state_jump = true;
    }
    else if (last_lio_stable_state_ready_)
    {
      lio_state_jump_trans_m = (_state.pos_end - last_lio_stable_state_.pos_end).norm();
      const Eigen::Matrix3d delta_rot = last_lio_stable_state_.rot_end.transpose() * _state.rot_end;
      lio_state_jump_rot_deg = Eigen::AngleAxisd(delta_rot).angle() * 57.29577951308232;
      lio_state_jump = (lio_state_jump_max_trans_m_ > 0.0 && lio_state_jump_trans_m > lio_state_jump_max_trans_m_) ||
                       (lio_state_jump_max_rot_deg_ > 0.0 && lio_state_jump_rot_deg > lio_state_jump_max_rot_deg_);
    }
  }
  if (lio_state_jump)
  {
    ROS_WARN_THROTTLE(1.0,
                      "[LIO] Reject state jump: dtrans=%.3f/%.3f m, drot=%.3f/%.3f deg, finite=%d.",
                      lio_state_jump_trans_m,
                      lio_state_jump_max_trans_m_,
                      lio_state_jump_rot_deg,
                      lio_state_jump_max_rot_deg_,
                      static_cast<int>(lio_pose_finite));
  }

  const bool lio_unstable_after_state = voxelmap_manager->isLidarDegenerated() || lio_state_jump;
  if (lio_unstable_after_state)
  {
    lio_degenerate_frame_count_++;
  }
  else
  {
    lio_degenerate_frame_count_ = 0;
    lio_freeze_state_ready_ = false;
    last_lio_stable_state_ = _state;
    last_lio_stable_state_ready_ = true;
  }
  if (lio_freeze_state_when_degenerate_ &&
      lio_unstable_after_state &&
      lio_degenerate_frame_count_ >= lio_freeze_degenerate_min_frames_)
  {
    if (!lio_freeze_state_ready_)
    {
      lio_freeze_state_ = last_lio_stable_state_ready_ ? last_lio_stable_state_ :
                          (ekf_finish_once ? latest_ekf_state : _state);
      lio_freeze_state_.vel_end.setZero();
      lio_freeze_state_ready_ = true;
    }
    _state = lio_freeze_state_;
    _state.vel_end.setZero();
    applyDegeneracyGuardCorrections("LIO-freeze");
    snapStateForDeterminism(_state);
    voxelmap_manager->state_ = _state;
    if (vio_manager) vio_manager->updateFrameState(_state);
    ROS_WARN_THROTTLE(1.0,
                      "[LIO] Freeze state during unstable LiDAR constraints: frames=%d ratio=%.6f.",
                      lio_degenerate_frame_count_,
                      voxelmap_manager->getLidarConstraintRatio());
  }
  applyUwbUpdate("LIO");
  advanceUwbOutputCorrection();
  if (!validateStateForSafety("LIO-after-uwb", state_before_lio_update, _state, false, false))
  {
    if (vio_manager) vio_manager->updateFrameState(_state);
    return;
  }

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en && !(safety_guard_enable_ && safety_fail_safe_mode_))
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(save_path + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(save_path + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    const V3D out_pos = outputPosition();
    evoFile << LidarMeasures.last_lio_update_time << " " << out_pos[0] << " " << out_pos[1] << " " << out_pos[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  if ((!update_decision_ready_ || current_decision_.allow_publish) &&
      validateStateForSafety("LIO-pre-odom", state_before_lio_update, _state, false, false))
  {
    publish_odometry(pubOdomAftMapped);
  }

  double t3 = omp_get_wtime();

  const int map_update_stride = std::max(1, lio_map_update_stride_);
  lio_map_update_counter_++;
  lio_frames_since_voxel_map_update_++;
  const bool lio_degenerated = voxelmap_manager->isLidarDegenerated() || lio_freeze_state_ready_;
  const bool stride_ready = (map_update_stride <= 1) || ((lio_map_update_counter_ % map_update_stride) == 0);
  const double time_since_map_update =
      (lio_last_voxel_map_update_time_ >= 0.0 && LidarMeasures.last_lio_update_time > lio_last_voxel_map_update_time_) ?
      (LidarMeasures.last_lio_update_time - lio_last_voxel_map_update_time_) : std::numeric_limits<double>::infinity();
	  const bool force_map_context = lio_force_voxel_map_update_;
  const bool force_by_frames =
      lio_force_voxel_map_update_ && force_map_context &&
      lio_frames_since_voxel_map_update_ >= std::max(1, lio_force_map_update_lidar_frames_);
	  const bool force_by_time =
	      lio_force_voxel_map_update_ && force_map_context && lio_force_map_update_interval_ > 0.0 &&
	      time_since_map_update >= lio_force_map_update_interval_;
	  const bool force_ready = force_by_frames || force_by_time;
	  MapUpdateDecision map_decision =
	      decideMapUpdate(lio_degenerated, stride_ready, force_ready, true);
	  bool do_map_update = map_decision.allow;
	  std::string voxel_map_skip_reason = map_decision.skip_reason;
	  if (!do_map_update && map_decision.reason == RejectReason::SMALL_MOTION && force_ready)
  {
    do_map_update = true;
    voxel_map_skip_reason = force_by_frames ? "forced_by_lidar_frames" : "forced_by_time";
    ROS_WARN_THROTTLE(1.0,
	                      "[LIO] Force voxel map update: reason=%s frames_since=%d time_since=%.3f s.",
                      voxel_map_skip_reason.c_str(),
                      lio_frames_since_voxel_map_update_,
                      time_since_map_update);
  }
  deg_guard_last_lio_voxel_map_updated_ = do_map_update;
  deg_guard_last_lio_voxel_map_skip_reason_ = voxel_map_skip_reason;
  double t4 = t3;
  if (!do_map_update)
  {
    ROS_WARN_THROTTLE(1.0,
                      "[LIO] Skip voxel map update: reason=%s degenerated=%d stride_ready=%d frames_since=%d time_since=%.3f s.",
                      voxel_map_skip_reason.c_str(),
                      static_cast<int>(lio_degenerated),
                      static_cast<int>(stride_ready),
                      lio_frames_since_voxel_map_update_,
                      time_since_map_update);
  }

	  if (do_map_update)
	  {
	    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
    transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
    const size_t update_count = std::min({world_lidar->points.size(),
                                          voxelmap_manager->pv_list_.size(),
                                          voxelmap_manager->cross_mat_list_.size(),
                                          voxelmap_manager->body_cov_list_.size()});
    if (update_count == 0)
    {
	      do_map_update = false;
	      deg_guard_last_lio_voxel_map_updated_ = false;
	      deg_guard_last_lio_voxel_map_skip_reason_ = "NO_POINTS";
	      ROS_WARN_THROTTLE(1.0, "[LIO] Skip voxel map update: reason=NO_POINTS.");
    }
    else
    {
      if (update_count < world_lidar->points.size() ||
          update_count < voxelmap_manager->pv_list_.size())
      {
        ROS_WARN_THROTTLE(1.0,
                          "[LIO] Voxel map update point count mismatch: world=%zu pv=%zu cross=%zu cov=%zu, update=%zu.",
                          world_lidar->points.size(),
                          voxelmap_manager->pv_list_.size(),
                          voxelmap_manager->cross_mat_list_.size(),
                          voxelmap_manager->body_cov_list_.size(),
                          update_count);
      }
      for (size_t i = 0; i < update_count; i++)
      {
        voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
        M3D var = voxelmap_manager->body_cov_list_[i];
        var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
              (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
        voxelmap_manager->pv_list_[i].var = var;
      }
      voxelmap_manager->pv_list_.resize(update_count);
      voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
      lio_frames_since_voxel_map_update_ = 0;
      lio_last_voxel_map_update_time_ = LidarMeasures.last_lio_update_time;
      if (print_console_timing_en_ && (frame_num % std::max(1, print_console_timing_stride_) == 0))
      {
        std::cout << "[ LIO ] Update Voxel Map, reason=" << voxel_map_skip_reason << std::endl;
      }
      _pv_list = voxelmap_manager->pv_list_;

      t4 = omp_get_wtime();

      if (voxelmap_manager->config_setting_.map_sliding_en)
      {
        voxelmap_manager->mapSliding();
	      }
	    }
	  }
		  if (local_mode_ == "DEGRADED_BOOTSTRAP" || local_mode_ == "LOCAL_REINIT" || local_mode_ == "RELOCALIZE")
	  {
	    lio_bootstrap_frames_++;
	  }
	  updateLocalTrackingLostDetectors("LIO");

	  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  if ((!update_decision_ready_ || current_decision_.allow_publish) &&
      !(safety_guard_enable_ && safety_fail_safe_mode_) &&
      validateStateForSafety("LIO-pre-publish", state_before_lio_update, _state, false, false))
  {
    if (!img_en) publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
    if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
    if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
    publish_path(pubPath);
    publish_mavros(mavros_pose_publisher);
  }
  if (deg_guard_last_lio_update_executed_ &&
      deg_guard_last_update_status_ != "skipped" &&
      deg_guard_last_update_status_ != "rejected")
  {
    recordReliableStateForSafety("LIO");
  }

  double t5 = omp_get_wtime();

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  const bool print_console_timing = print_console_timing_en_ &&
                                    ((frame_num % std::max(1, print_console_timing_stride_)) == 0);
  if (print_console_timing)
  {
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "postProcess+Publish", t5 - t4);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t5 - t0);
    printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  }

  if (vio_manager)
  {
    auto formatDouble6 = [](double value)
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6) << value;
      return oss.str();
    };

    auto makeTableRow = [](const std::string &left, const std::string &right)
    {
      std::ostringstream oss;
      oss << "| " << std::left << std::setw(29) << left
          << " | " << std::left << std::setw(27) << right << " |";
      return oss.str();
    };

    const double lio_total_time = t5 - t0;
    std::vector<std::string> lines;
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back("|                         LIO Mapping Time                    |");
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("Algorithm Stage", "Time (secs)"));
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("DownSample", formatDouble6(t_down - t0)));
    lines.push_back(makeTableRow("ICP", formatDouble6(t2 - t1)));
    lines.push_back(makeTableRow("updateVoxelMap", formatDouble6(t4 - t3)));
    lines.push_back(makeTableRow("postProcess+Publish", formatDouble6(t5 - t4)));
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("Current Total Time", formatDouble6(lio_total_time)));
    lines.push_back(makeTableRow("Average Total Time", formatDouble6(aver_time_consu)));
    lines.push_back(makeTableRow("Budget (s)", formatDouble6(frame_time_budget_s_)));
    lines.push_back("+-------------------------------------------------------------+");
    vio_manager->appendTimingLogLines(lines);
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  const V3D out_pos = outputPosition();
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << out_pos.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    //std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    //std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    string all_points_dir(save_path + string("map_dense.pcd"));
    string downsampled_points_dir(save_path + string("map.pcd"));
    string downsampled_points_dir2(save_path + string("pose.pcd"));
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
 
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the raw point cloud data
      pcd_writer.writeBinary(downsampled_points_dir2, *downsampled_cloud); 
      pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);

      std::cout << GREEN << "All point cloud data saved to: " << all_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "All point cloud data saved to: " << all_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::print_landmarks()
{
  if (vio_manager->board_world_flag_.empty())
  {
    std::cout << YELLOW << "[Aruco] No board entries to print." << RESET << std::endl;
    return;
  }

  std::cout << YELLOW << "[Aruco] Final board first-observation positions:" << RESET << std::endl;

  for (const auto& item : vio_manager->board_world_flag_)
  {
    const int id = item.first;
    const bool initialized = item.second;

    auto pos_it = vio_manager->board_world_positions_.find(id);
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    if (pos_it != vio_manager->board_world_positions_.end())
    {
      position = pos_it->second;
    }

    if (initialized)
    {
      std::cout << YELLOW << "  [INIT] Board " << id
                << " -> (" << position.x() << ", "
                << position.y() << ", " << position.z() << ")"
                << RESET << std::endl;
    }
    else
    {
      std::cout << YELLOW << "  [UNINIT] Board " << id
                << " -> not observed yet"
                << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  auto formatDouble6 = [](double value)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
  };

  auto makeTableRow = [](const std::string &left, const std::string &right)
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(29) << left
        << " | " << std::left << std::setw(27) << right << " |";
    return oss.str();
  };

  ros::Rate rate(5000);
  int i = 0;
  double sum = 0, t = 0;
  while (ros::ok()) 
  {
    const double t1 = omp_get_wtime();
    ros::spinOnce();
    const double t_spin_end = omp_get_wtime();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    const double t_sync_end = omp_get_wtime();
    handleFirstFrame();

    processImu();
    const double t_imu_end = omp_get_wtime();
    if (skip_mapping_this_frame_)
    {
      writeDegeneracyGuardLog("imu-init-skip");
      rate.sleep();
      continue;
    }

    // if (!p_imu->imu_time_init) continue;

    const EKF_STATE frame_mode = LidarMeasures.lio_vio_flg;
    stateEstimationAndMapping();

    const double t2 = omp_get_wtime();
    const double frame_time = t2 - t1;
    updateRuntimeGuard(frame_time);

    if (vio_manager)
    {
      std::vector<std::string> lines;
      const double t_spin = t_spin_end - t1;
      const double t_sync = t_sync_end - t_spin_end;
      const double t_imu = t_imu_end - t_sync_end;
      const double t_mapping = t2 - t_imu_end;
      const double t_other = std::max(0.0, frame_time - (t_spin + t_sync + t_imu + t_mapping));

      std::string mode_str = "WAIT";
      if (frame_mode == VIO) mode_str = "VIO";
      else if (frame_mode == LIO) mode_str = "LIO";
      else if (frame_mode == LO) mode_str = "LO";

      std::ostringstream line;
      line << "[ Frame ] idx=" << (i + 1)
           << ", mode=" << mode_str
           << ", current=" << formatDouble6(frame_time)
           << " s, budget=" << formatDouble6(frame_time_budget_s_);
      lines.push_back(line.str());

      std::ostringstream line_cost;
      line_cost << "[ Frame Cost ] spin=" << formatDouble6(t_spin)
                << ", sync=" << formatDouble6(t_sync)
                << ", imu=" << formatDouble6(t_imu)
                << ", mapping=" << formatDouble6(t_mapping)
                << ", other=" << formatDouble6(t_other);
      lines.push_back(line_cost.str());

      vio_manager->appendTimingLogLines(lines);
    }

    i ++;
    t += frame_time;
    if (i % 2 == 0)
    {
      sum += t;
      const bool print_console_timing = print_console_timing_en_ &&
                                        (((i / 2) % std::max(1, print_console_timing_stride_)) == 0);
      if (print_console_timing)
      {
        printf("\033[1;45m+-------------------------------------------------------------+\033[0m\n");
        printf("\033[1;95m| %-29s | %-27d |\033[0m\n", "Frame number", i/2);
        printf("\033[1;95m| %-29s | %-27f |\033[0m\n", "Current frame time", t);
        printf("\033[1;95m| %-29s | %-27f |\033[0m\n", "Average frame time", sum / (i/2));
        printf("\033[1;45m+-------------------------------------------------------------+\033[0m\n");
      }

      if (vio_manager)
      {
        std::vector<std::string> lines;
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back("|                         Frame Time                          |");
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back(makeTableRow("Frame number", std::to_string(i / 2)));
        lines.push_back(makeTableRow("Current frame time", formatDouble6(t)));
        lines.push_back(makeTableRow("Average frame time", formatDouble6(sum / (i / 2))));
        lines.push_back(makeTableRow("Budget (s)", formatDouble6(frame_time_budget_s_)));
        lines.push_back("+-------------------------------------------------------------+");
        vio_manager->appendTimingLogLines(lines);
      }

      t = 0;
    }
  }
  savePCD();
  if (aruco_landmarks_en) print_landmarks();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    if (deterministic_prop_imu_buffer_sort_en_)
    {
      std::sort(prop_imu_buffer.begin(), prop_imu_buffer.end(),
                [](const sensor_msgs::Imu &a, const sensor_msgs::Imu &b) {
                  return a.header.stamp.toSec() < b.header.stamp.toSec();
                });
    }
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  if (trans_cloud == nullptr) return;
  PointCloudXYZI().swap(*trans_cloud);
  if (input_cloud == nullptr || input_cloud->empty()) return;
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  while (max_lidar_buffer_size_ > 0 && static_cast<int>(lid_raw_data_buffer.size()) > max_lidar_buffer_size_)
  {
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
  }
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO_THROTTLE(1.0, "Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  while (max_lidar_buffer_size_ > 0 && static_cast<int>(lid_raw_data_buffer.size()) > max_lidar_buffer_size_)
  {
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
  }
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (last_timestamp_lidar > 0.0 && fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    if (!deterministic_imu_accept_out_of_order_en_)
    {
      ROS_ERROR("imu loop back. \n");
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      return;
    }
    ROS_WARN("imu loop back, offset: %lf, inserting anyway\n", last_timestamp_imu - timestamp);
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  if (deterministic_imu_accept_out_of_order_en_)
  {
    if (timestamp > last_timestamp_imu) last_timestamp_imu = timestamp;
  }
  else
  {
    last_timestamp_imu = timestamp;
  }

  imu_buffer.push_back(msg);
  while (max_imu_buffer_size_ > 0 && static_cast<int>(imu_buffer.size()) > max_imu_buffer_size_)
  {
    imu_buffer.pop_front();
  }
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init)
    {
      prop_imu_buffer.push_back(*msg);
      while (max_prop_imu_buffer_size_ > 0 && static_cast<int>(prop_imu_buffer.size()) > max_prop_imu_buffer_size_)
      {
        prop_imu_buffer.pop_front();
      }
    }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (!deterministic_image_buffer_sort_en_)
  {
    if (std::fabs(msg_header_time - last_timestamp_img) < 0.001) return;
    if (msg_header_time < last_timestamp_img)
    {
      ROS_ERROR("image loop back. \n");
      return;
    }
  }
  ROS_INFO_THROTTLE(1.0, "Get image, its header time: %.6f", msg_header_time);

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (deterministic_image_buffer_sort_en_)
  {
    for (const double buffered_time : img_time_buffer)
    {
      if (std::fabs(buffered_time - img_time_correct) < 0.001)
      {
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
      }
    }
  }
  else if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  if (deterministic_image_buffer_sort_en_)
  {
    const auto insert_it = std::lower_bound(img_time_buffer.begin(), img_time_buffer.end(), img_time_correct);
    const auto insert_idx = std::distance(img_time_buffer.begin(), insert_it);
    img_time_buffer.insert(insert_it, img_time_correct);
    img_buffer.insert(img_buffer.begin() + insert_idx, img_cur);
  }
  else
  {
    img_buffer.push_back(img_cur);
    img_time_buffer.push_back(img_time_correct);
  }
  while (max_img_buffer_size_ > 0 && static_cast<int>(img_buffer.size()) > max_img_buffer_size_)
  {
    img_buffer.pop_front();
    img_time_buffer.pop_front();
  }

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  if (deterministic_image_buffer_sort_en_)
  {
    if (img_time_correct > last_timestamp_img) last_timestamp_img = img_time_correct;
  }
  else
  {
    last_timestamp_img = img_time_correct;
  }
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  const bool pending_livo_vio =
      deterministic_pending_vio_image_en_ && slam_mode_ == LIVO && meas.lio_vio_flg == LIO && has_pending_vio_img_;
  if (lid_raw_data_buffer.empty() && lidar_en && !pending_livo_vio) return false;
  if (img_en && img_buffer.empty() && !pending_livo_vio) return false;
  if (imu_buffer.empty() && imu_en && !pending_livo_vio) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    // 确保 imu_buffer 按时间戳有序，消除乱序送达导致的 draining 非确定性
    if (deterministic_imu_buffer_sort_en_)
    {
      std::sort(imu_buffer.begin(), imu_buffer.end(),
                [](const sensor_msgs::Imu::ConstPtr &a, const sensor_msgs::Imu::ConstPtr &b) {
                  return a->header.stamp.toSec() < b->header.stamp.toSec();
                });
    }
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    // 确保 IMU 按时间戳严格有序，消除回调乱序导致的非确定性
    if (deterministic_imu_buffer_sort_en_)
    {
      std::sort(m.imu.begin(), m.imu.end(),
                [](const sensor_msgs::Imu::ConstPtr &a, const sensor_msgs::Imu::ConstPtr &b) {
                  return a->header.stamp.toSec() < b->header.stamp.toSec();
                });
    }

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      if (deterministic_sync_wait_for_image_lookahead_en_ &&
          static_cast<int>(img_time_buffer.size()) < sync_img_buffer_min_size_)
      {
        return false;
      }
      if (deterministic_sync_wait_for_image_lookahead_en_ &&
          sync_img_lookahead_time_ > 0.0 &&
          img_time_buffer.back() < img_time_buffer.front() + sync_img_lookahead_time_)
      {
        return false;
      }
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = last_timestamp_imu;

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      // 确保 imu_buffer 按时间戳有序，消除乱序送达导致的 draining 非确定性
      if (deterministic_imu_buffer_sort_en_)
      {
        std::sort(imu_buffer.begin(), imu_buffer.end(),
                  [](const sensor_msgs::Imu::ConstPtr &a, const sensor_msgs::Imu::ConstPtr &b) {
                    return a->header.stamp.toSec() < b->header.stamp.toSec();
                  });
      }
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      // 确保 IMU 按时间戳严格有序，消除回调乱序导致的非确定性
      if (deterministic_imu_buffer_sort_en_)
      {
        std::sort(m.imu.begin(), m.imu.end(),
                  [](const sensor_msgs::Imu::ConstPtr &a, const sensor_msgs::Imu::ConstPtr &b) {
                    return a->header.stamp.toSec() < b->header.stamp.toSec();
                  });
      }

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      if (deterministic_pending_vio_image_en_)
      {
        pending_vio_img_ = img_buffer.front();
        pending_vio_time_ = img_capture_time;
        has_pending_vio_img_ = true;
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        if (deterministic_sync_wait_for_image_lookahead_en_ &&
            (static_cast<int>(img_buffer.size()) < sync_img_buffer_min_size_ ||
             (sync_img_lookahead_time_ > 0.0 &&
              !img_time_buffer.empty() &&
              img_time_buffer.back() < img_time_buffer.front() + sync_img_lookahead_time_)))
        {
          sig_buffer.notify_all();
        }
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      if (deterministic_pending_vio_image_en_ && !has_pending_vio_img_) return false;
      if (!deterministic_pending_vio_image_en_ && img_buffer.empty()) return false;
      double img_capture_time =
          deterministic_pending_vio_image_en_ ? pending_vio_time_ : img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = deterministic_pending_vio_image_en_ ? pending_vio_img_ : img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      if (deterministic_pending_vio_image_en_)
      {
        pending_vio_img_.release();
        pending_vio_time_ = 0.0;
        has_pending_vio_img_ = false;
      }
      else
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes,const ros::Publisher &pubLaserCloudMap, VIOManagerPtr vio_manager)
{

  if (safety_guard_enable_)
  {
    std::string reason;
    if (safety_fail_safe_mode_ || !isStateFiniteForSafety(_state, "publish_frame_world", &reason))
    {
      if (!safety_fail_safe_mode_) enterFailSafe(reason.empty() ? "publish_frame_world_invalid_state" : reason, nullptr);
      return;
    }
  }
  if (pcl_w_wait_pub == nullptr || pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  const bool can_colorize = (vio_manager != nullptr) &&
                            (vio_manager->new_frame_ != nullptr) &&
                            (vio_manager->new_frame_->cam_ != nullptr) &&
                            !vio_manager->img_rgb.empty();
  const bool need_rgb_cloud = img_en && (colorize_cloud_en_ || pcd_save_en) && can_colorize;
  if (need_rgb_cloud)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }
  }
  else
  {
    PointCloudXYZI().swap(*pcl_wait_pub);
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (need_rgb_cloud)
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    int size = feats_undistort ? feats_undistort->points.size() : 0;
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;
    
    if (need_rgb_cloud)
    {
      //global map

      if (global_map_pub)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_wait_save_filter(new pcl::PointCloud<pcl::PointXYZRGB>);  
        pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilterMap;
        downSizeFilterMap.setInputCloud(laserCloudWorldRGB);  //当前帧
        //downSizeFilterMap.setInputCloud(pcl_wait_save);     //整个地图
        downSizeFilterMap.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
        downSizeFilterMap.filter(*pcl_wait_save_filter);
        
        //*pcl_wait_save += *laserCloudWorldRGB;          //总地图添加未过滤点云
        *pcl_wait_save += *pcl_wait_save_filter;        //添加过滤后的点云

        pcl::toROSMsg(*pcl_wait_save, laserCloudmsg);   //发布
        pubLaserCloudMap.publish(laserCloudmsg);
      }
      else
      {
        *pcl_wait_save += *laserCloudWorldRGB;
      }
      if (pcd_cache_max_points > 0 && pcl_wait_save->size() > static_cast<size_t>(pcd_cache_max_points))
      {
        size_t overflow = pcl_wait_save->size() - static_cast<size_t>(pcd_cache_max_points);
        pcl_wait_save->points.erase(pcl_wait_save->points.begin(), pcl_wait_save->points.begin() + overflow);
        pcl_wait_save->width = pcl_wait_save->points.size();
        pcl_wait_save->height = 1;
        pcl_wait_save->is_dense = false;
      }
    }
    else
    {
      if (global_map_pub)
      {
        pcl::PointCloud<PointType>::Ptr pcl_wait_save_filter(new pcl::PointCloud<PointType>);
        pcl::VoxelGrid<PointType> downSizeFilterMap;
        downSizeFilterMap.setInputCloud(pcl_w_wait_pub);  //当前帧
        //downSizeFilterMap.setInputCloud(pcl_wait_save);     //整个地图
        downSizeFilterMap.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
        downSizeFilterMap.filter(*pcl_wait_save_filter);
        
        //*pcl_wait_save_intensity += *pcl_w_wait_pub;          //总地图添加未过滤点云
        *pcl_wait_save_intensity += *pcl_wait_save_filter;        //添加过滤后的点云

        pcl::toROSMsg(*pcl_wait_save_intensity, laserCloudmsg);   //发布
        pubLaserCloudMap.publish(laserCloudmsg);
      }
      else
      {
        *pcl_wait_save_intensity += *pcl_w_wait_pub;
      }
      if (pcd_cache_max_points > 0 && pcl_wait_save_intensity->size() > static_cast<size_t>(pcd_cache_max_points))
      {
        size_t overflow = pcl_wait_save_intensity->size() - static_cast<size_t>(pcd_cache_max_points);
        pcl_wait_save_intensity->points.erase(pcl_wait_save_intensity->points.begin(), pcl_wait_save_intensity->points.begin() + overflow);
        pcl_wait_save_intensity->width = pcl_wait_save_intensity->points.size();
        pcl_wait_save_intensity->height = 1;
        pcl_wait_save_intensity->is_dense = false;
      }
    }

    scan_wait_num++;
    
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      //string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      string all_points_dir(save_path + to_string(pcd_index) + string(".pcd"));
      /*string all_points_dir(save_path + string("map_dense.pcd"));
      string downsampled_points_dir(save_path + string("map.pcd"));
      string downsampled_points_dir2(save_path + string("pose.pcd"));*/
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en)
        {
          /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
          voxel_filter.setInputCloud(pcl_wait_save);
          voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
          voxel_filter.filter(*downsampled_cloud);*/

          //pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the raw point cloud data
          //pcd_writer.writeBinary(downsampled_points_dir2, *downsampled_cloud); 
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);    //清空缓存
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }        
        scan_wait_num = 0;
      }
    }
  }

  if (save_log_en)
  {
    Eigen::Quaterniond q(_state.rot_end);
    const V3D out_pos = outputPosition();
    std::ostringstream stamp_oss;
    stamp_oss << std::fixed << std::setprecision(9) << LidarMeasures.last_lio_update_time;
    fout_pcd_pos << stamp_oss.str() << " " << out_pos[0] << " " << out_pos[1] << " " << out_pos[2] << " "
                 << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << endl;
  }
  
  if (need_rgb_cloud && laserCloudWorldRGB->size() > 0) PointCloudXYZI().swap(*pcl_wait_pub);
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  if (safety_guard_enable_ && safety_fail_safe_mode_) return;
  if (visual_sub_map == nullptr) return;
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  if (safety_guard_enable_ && safety_fail_safe_mode_) return;
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  const V3D out_pos = outputPosition();
  out.position.x = out_pos(0);
  out.position.y = out_pos(1);
  out.position.z = out_pos(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  if (safety_guard_enable_)
  {
    std::string reason;
    if (safety_fail_safe_mode_ || !isStateFiniteForSafety(_state, "publish_odom", &reason))
    {
      if (!safety_fail_safe_mode_) enterFailSafe(reason.empty() ? "publish_odom_invalid_state" : reason, nullptr);
      return;
    }
  }
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  const V3D out_pos = outputPosition();
  transform.setOrigin(tf::Vector3(out_pos(0), out_pos(1), out_pos(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
  sendUdpPose(out_pos);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  if (safety_guard_enable_)
  {
    std::string reason;
    if (safety_fail_safe_mode_ || !isStateFiniteForSafety(_state, "publish_mavros", &reason))
    {
      if (!safety_fail_safe_mode_) enterFailSafe(reason.empty() ? "publish_mavros_invalid_state" : reason, nullptr);
      return;
    }
  }
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  if (safety_guard_enable_)
  {
    std::string reason;
    if (safety_fail_safe_mode_ || !isStateFiniteForSafety(_state, "publish_path", &reason))
    {
      if (!safety_fail_safe_mode_) enterFailSafe(reason.empty() ? "publish_path_invalid_state" : reason, nullptr);
      return;
    }
  }
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
