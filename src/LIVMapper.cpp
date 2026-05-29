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
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <system_error>
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
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents(nh);
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper()
{
  if (udp_socket_fd_ >= 0)
  {
    ::close(udp_socket_fd_);
    udp_socket_fd_ = -1;
  }
}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
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
  IMG_POINT_COV = 200.0;  // 降低visual EKF权重：100.0→200.0，使EKF对visual测量的信任度减半
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<bool>("vio/visual_map_prune_en", visual_map_prune_en, true);
  nh.param<int>("vio/visual_map_max_voxels", visual_map_max_voxels, 12000);
  nh.param<int>("vio/visual_map_max_points_per_voxel", visual_map_max_points_per_voxel, 24);
  nh.param<int>("vio/visual_map_max_total_points", visual_map_max_total_points, 180000);
  nh.param<int>("vio/visual_map_max_add_per_frame", visual_map_max_add_per_frame_, 600);
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
  nh.param<double>("vio/visual_update_max_rot_deg", vio_visual_update_max_rot_deg_, 2.0);
  nh.param<double>("vio/visual_update_max_backward_m", vio_visual_update_max_backward_m_, 0.03);
  nh.param<double>("vio/visual_update_max_backward_ratio", vio_visual_update_max_backward_ratio_, 0.08);
  nh.param<double>("vio/visual_update_backward_abs_floor_m", vio_visual_update_backward_abs_floor_m_, 0.003);
  nh.param<double>("vio/visual_update_max_lateral_m", vio_visual_update_max_lateral_m_, 0.08);
  nh.param<double>("vio/visual_update_max_lateral_ratio", vio_visual_update_max_lateral_ratio_, 0.35);
  nh.param<double>("vio/visual_update_max_exposure_delta", vio_visual_update_max_exposure_delta_, 0.30);
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

  keyframe_trans_thresh_min_nominal_ = keyframe_trans_thresh_min_;
  keyframe_trans_thresh_max_nominal_ = keyframe_trans_thresh_max_;
  keyframe_rot_thresh_min_deg_nominal_ = keyframe_rot_thresh_min_deg_;
  keyframe_rot_thresh_max_deg_nominal_ = keyframe_rot_thresh_max_deg_;
  keyframe_max_skip_frames_nominal_ = keyframe_max_skip_frames_;
  vio_max_iterations_nominal_ = max_iterations;

  pub_scan_num = std::max(1, pub_scan_num);
  publish_img_stride_ = std::max(1, publish_img_stride_);
  lio_map_update_stride_ = std::max(1, lio_map_update_stride_);
  print_console_timing_stride_ = std::max(1, print_console_timing_stride_);

  pub_scan_num_nominal_ = pub_scan_num;
  dense_map_en_nominal_ = dense_map_en;
  pcd_save_en_nominal_ = pcd_save_en;
  colorize_cloud_en_nominal_ = colorize_cloud_en_;

  nh.param<bool>("runtime_guard/enable", runtime_guard_en_, true);
  nh.param<double>("runtime_guard/frame_time_budget_s", frame_time_budget_s_, frame_time_budget_s_);
  if (frame_time_budget_s_ <= 0.0)
  {
    ROS_WARN("[RuntimeGuard] Invalid frame_time_budget_s=%.6f, fallback to 0.100000 s", frame_time_budget_s_);
    frame_time_budget_s_ = 0.1;
  }
  ROS_INFO("[RuntimeGuard] enable=%d, frame_time_budget_s=%.6f s", static_cast<int>(runtime_guard_en_), frame_time_budget_s_);

  pub_scan_num_degraded_ = std::max(1, pub_scan_num_degraded_);
  runtime_over_budget_trigger_frames_ = std::max(1, runtime_over_budget_trigger_frames_);
  runtime_recover_trigger_frames_ = std::max(1, runtime_recover_trigger_frames_);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::updateRuntimeGuard(double frame_time_s)
{
  if (!runtime_guard_en_) return;

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
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->min_retrieve_points = vio_min_retrieve_points_;
  vio_manager->min_update_meas = vio_min_update_meas_;
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
  vio_manager->visual_update_max_backward_m = vio_visual_update_max_backward_m_;
  vio_manager->visual_update_max_backward_ratio = vio_visual_update_max_backward_ratio_;
  vio_manager->visual_update_backward_abs_floor_m = vio_visual_update_backward_abs_floor_m_;
  vio_manager->visual_update_max_lateral_m = vio_visual_update_max_lateral_m_;
  vio_manager->visual_update_max_lateral_ratio = vio_visual_update_max_lateral_ratio_;
  vio_manager->visual_update_max_exposure_delta = vio_visual_update_max_exposure_delta_;
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

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  snapStateForDeterminism(_state);
  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  static int vio_dispatch_count = 0;
  static int lio_dispatch_count = 0;

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
  snapStateForDeterminism(_state);
  voxelmap_manager->state_ = _state;
  if (state_update_flg) latest_ekf_state = _state;
}

void LIVMapper::handleVIO() 
{

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
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

    publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);

    euler_cur = RotMtoEuler(_state.rot_end);
    fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
             << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
             << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;

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

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);
  snapStateForDeterminism(_state);
  vio_manager->updateFrameState(_state);
  updateVisualObservationHints();

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

  publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
  if (!suppress_image_pub_)
  {
    publish_img_counter_++;
    if (publish_img_counter_ % std::max(1, publish_img_stride_) == 0)
    {
      publish_img_rgb(pubImage, vio_manager);
    }
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;

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

  if (!adaptive_visual_selector_en)
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
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
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

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;
  snapStateForDeterminism(_state);
  voxelmap_manager->state_ = _state;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
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
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  const int map_update_stride = std::max(1, lio_map_update_stride_);
  lio_map_update_counter_++;
  const bool do_map_update = (map_update_stride <= 1) || ((lio_map_update_counter_ % map_update_stride) == 0);
  double t4 = t3;

  if (do_map_update)
  {
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
    transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
    for (size_t i = 0; i < world_lidar->points.size(); i++)
    {
      voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
      M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
      M3D var = voxelmap_manager->body_cov_list_[i];
      var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
            (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
      voxelmap_manager->pv_list_[i].var = var;
    }
    voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
    if (print_console_timing_en_ && (frame_num % std::max(1, print_console_timing_stride_) == 0))
    {
      std::cout << "[ LIO ] Update Voxel Map" << std::endl;
    }
    _pv_list = voxelmap_manager->pv_list_;

    t4 = omp_get_wtime();

    if (voxelmap_manager->config_setting_.map_sliding_en)
    {
      voxelmap_manager->mapSliding();
    }
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, pubLaserCloudMap, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

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
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
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
  PointCloudXYZI().swap(*trans_cloud);
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

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
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

  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  const bool need_rgb_cloud = img_en && (colorize_cloud_en_ || pcd_save_en);
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
    int size = feats_undistort->points.size();
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
    fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                  << " " << q.z() << " " << endl;
  }
  
  if (need_rgb_cloud && laserCloudWorldRGB->size() > 0) PointCloudXYZI().swap(*pcl_wait_pub);
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
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
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
  sendUdpPose(_state.pos_end);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
