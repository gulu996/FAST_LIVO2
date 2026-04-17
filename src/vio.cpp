/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>

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

std::string formatDouble6(double value)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}

std::string makeTableRow(const std::string &left, const std::string &right)
{
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(29) << left
      << " | " << std::left << std::setw(27) << right << " |";
  return oss.str();
}
}

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  if (timing_log_file.is_open()) timing_log_file.close();
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO(ros::NodeHandle &nh)
{
  visual_submap = new SubSparseMap;

  // Initialize camera matrix
  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  cameraMatrix_ = (cv::Mat_<float>(3, 3) << fx, 0, cx,
                                            0, fy, cy,
                                            0,  0,  1);
  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  // Initialize distortion coefficients
  pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
  d0 = pinhole_cam->d0();
  d1 = pinhole_cam->d1();
  d2 = pinhole_cam->d2();
  d3 = pinhole_cam->d3();
  distCoeffs_ = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, 0);
  printf("distCoeffs: %.6lf, %.6lf, %.6lf, %.6lf\n", d0, d1, d2, d3);

  parameters_ = cv::aruco::DetectorParameters::create();
  dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

  //loadArucoWorldPositions
  // 检查参数是否存在
  if (!nh.hasParam("/aruco_landmarks") || !aruco_landmarks_en) 
  {
    if (!nh.hasParam("/aruco_landmarks")) ROS_WARN("[Aruco] No aruco_landmarks parameter found");
    if (!aruco_landmarks_en) ROS_INFO("[Aruco] Not Enable");
  }
  else
  {
    nh.param<double>("/aruco_landmarks/width", board_config_.width, 0.8);
    nh.param<double>("/aruco_landmarks/height", board_config_.height, 0.6);
    nh.param<double>("/aruco_landmarks/marker_size", marker_size, 0.16);
    nh.param<double>("/aruco_landmarks/delta_width_qr_center", board_config_.delta_width_qr_center, 0.28);
    nh.param<double>("/aruco_landmarks/delta_height_qr_center", board_config_.delta_height_qr_center, 0.18);
    aruco_min_quad_area_px = 300.0;
    aruco_pair_distance_rel_tol = 0.2;
    aruco_max_normal_diff_deg = 15.0;
    aruco_max_marker_depth_diff = 0.6;
    aruco_min_marker_depth = 0.1;
    aruco_max_marker_depth = 10.0;
    aruco_max_position_residual = 0.6;
    aruco_max_orientation_residual_deg = 25.0;
    aruco_position_noise_base = 0.01;
    aruco_orientation_noise_base = 0.1;
    aruco_process_stride = 4;
    aruco_use_orientation_update = false;
    aruco_normal_gate_deg = 35.0;
    aruco_update_max_rot_step_deg = 1.0;
    aruco_update_max_trans_step_m = 0.08;

    aruco_relative_positions_[1] = Eigen::Vector3d(-board_config_.delta_width_qr_center, board_config_.delta_height_qr_center, 0);   // 左上
    aruco_relative_positions_[2] = Eigen::Vector3d(board_config_.delta_width_qr_center, board_config_.delta_height_qr_center, 0);    // 右上
    aruco_relative_positions_[3] = Eigen::Vector3d(-board_config_.delta_width_qr_center, -board_config_.delta_height_qr_center, 0);  // 左下
    aruco_relative_positions_[4] = Eigen::Vector3d(board_config_.delta_width_qr_center, -board_config_.delta_height_qr_center, 0);   // 右下

    ROS_INFO("[Aruco] Marker size: %.3f meters", marker_size);
    ROS_INFO("[Aruco] Same-ID gating: area>=%.1f px^2, pair tol<=%.2f, normal<=%.1f deg",
         aruco_min_quad_area_px, aruco_pair_distance_rel_tol, aruco_max_normal_diff_deg);
    // 清空现有地标
    board_world_positions_.clear();
    board_world_orientations_.clear();
    board_world_flag_.clear();

    // 使用列表格式
    XmlRpc::XmlRpcValue markers_list;
    if (nh.getParam("/aruco_landmarks/markers", markers_list) && markers_list.getType() == XmlRpc::XmlRpcValue::TypeArray) 
    {
      ROS_INFO("[Aruco] Loading landmarks from list format");
      for (int i = 0; i < markers_list.size(); i++) 
      {
        XmlRpc::XmlRpcValue marker = markers_list[i];
        if (!marker.hasMember("id")) continue;

        int id = static_cast<int>(marker["id"]);
        int flag = 0;
        if (marker.hasMember("flag"))
        {
          flag = static_cast<int>(marker["flag"]);
        }

        board_world_orientations_[id] = Eigen::Matrix3d::Identity();

        if (flag)
        {
          if (!marker.hasMember("position"))
          {
            board_world_flag_[id] = false;
            board_world_positions_[id] = Eigen::Vector3d::Zero();
            ROS_WARN("[Aruco] Marker %d has flag=1 but no position, fallback to uninitialized.", id);
            continue;
          }

          XmlRpc::XmlRpcValue pos = marker["position"];
          if (pos.size() != 3)
          {
            board_world_flag_[id] = false;
            board_world_positions_[id] = Eigen::Vector3d::Zero();
            ROS_WARN("[Aruco] Marker %d has invalid position size, fallback to uninitialized.", id);
            continue;
          }

          board_world_flag_[id] = true;
          Eigen::Vector3d position(
            static_cast<double>(pos[0]),
            static_cast<double>(pos[1]),
            static_cast<double>(pos[2])
          );
          board_world_positions_[id] = position;
          ROS_INFO("[Aruco] Loaded marker %d at [%.2f, %.2f, %.2f]", id, position.x(), position.y(), position.z());
        }
        else
        {
          board_world_flag_[id] = false;
          board_world_positions_[id] = Eigen::Vector3d::Zero();
          ROS_INFO("[Aruco] Created marker %d as uninitialized", id);
        }
      }
    }    
  }

  width = cam->width();
  height = cam->height();
  image_resize_factor = cam->scale();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }

  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::initializeTimingLogFileIfNeeded()
{
  if (!timing_log_enable || timing_log_ready) return;
  if (timing_log_dir.empty()) return;

  std::string dir = timing_log_dir;
  if (dir.back() != '/') dir += '/';

  timing_log_file_path = dir + currentTimeForFilename() + ".log";
  timing_log_file.open(timing_log_file_path, std::ios::out | std::ios::app);
  if (!timing_log_file.is_open())
  {
    printf("[ VIO ] Warning: failed to open timing log file: %s\n", timing_log_file_path.c_str());
    timing_log_enable = false;
    return;
  }

  timing_log_ready = true;
  printf("[ VIO ] Timing log file: %s\n", timing_log_file_path.c_str());
}

void VIOManager::appendTimingLogLines(const vector<string> &lines)
{
  if (!timing_log_enable || lines.empty()) return;
  if (!timing_log_ready) initializeTimingLogFileIfNeeded();
  if (!timing_log_ready || !timing_log_file.is_open()) return;

  for (const auto &line : lines)
  {
    timing_log_file << line << "\n";
  }
  timing_log_file << "\n";
  timing_log_pending_frames++;
  if (timing_log_flush_stride <= 1 || timing_log_pending_frames >= timing_log_flush_stride)
  {
    timing_log_file.flush();
    timing_log_pending_frames = 0;
  }
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

bool VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  if (pt_new == nullptr) return false;

  if (visual_voxel_size <= 1e-6)
  {
    delete pt_new;
    return false;
  }

  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  if (!std::isfinite(pt_w[0]) || !std::isfinite(pt_w[1]) || !std::isfinite(pt_w[2]))
  {
    delete pt_new;
    return false;
  }

  const double voxel_size = visual_voxel_size;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    std::vector<VisualPoint *> &pts = iter->second->voxel_points;
    if (visual_map_max_points_per_voxel > 0 &&
        pts.size() >= static_cast<size_t>(visual_map_max_points_per_voxel))
    {
      delete pt_new;
      return false;
    }
    pts.push_back(pt_new);
    // Do not erase here: tracked pointers in current visual_submap may still
    // reference points in this voxel during the same frame.
    iter->second->count = pts.size();
    return true;
  }
  else
  {
    if (visual_map_max_voxels > 0 &&
        feat_map.size() >= static_cast<size_t>(visual_map_max_voxels))
    {
      delete pt_new;
      return false;
    }

    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    ot->count = ot->voxel_points.size();
    feat_map[position] = ot;
    return true;
  }
}

size_t VIOManager::getVisualPointCount() const
{
  size_t total = 0;
  for (const auto &kv : feat_map)
  {
    if (kv.second != nullptr) total += kv.second->voxel_points.size();
  }
  return total;
}

void VIOManager::pruneVisualMap()
{
  if (!visual_map_prune_en || feat_map.empty()) return;

  size_t removed_points = 0;
  size_t removed_voxels = 0;

  if (visual_map_max_points_per_voxel > 0)
  {
    for (auto &kv : feat_map)
    {
      VOXEL_POINTS *voxel = kv.second;
      if (voxel == nullptr) continue;
      auto &pts = voxel->voxel_points;
      while (pts.size() > static_cast<size_t>(visual_map_max_points_per_voxel))
      {
        delete pts.front();
        pts.erase(pts.begin());
        ++removed_points;
      }
      voxel->count = pts.size();
    }
  }

  size_t total_points = getVisualPointCount();
  const bool over_voxel_limit = visual_map_max_voxels > 0 && feat_map.size() > static_cast<size_t>(visual_map_max_voxels);
  const bool over_point_limit = visual_map_max_total_points > 0 && total_points > static_cast<size_t>(visual_map_max_total_points);
  if (!over_voxel_limit && !over_point_limit)
  {
    if (removed_points > 0)
    {
      printf("[ VIO ] Prune visual map: remove %zu points, voxels=%zu, points=%zu\n", removed_points, feat_map.size(), total_points);
    }
    return;
  }

  V3D center(0, 0, 0);
  if (state != nullptr) center = state->pos_end;

  std::vector<std::pair<double, VOXEL_LOCATION>> ordered_voxels;
  ordered_voxels.reserve(feat_map.size());
  for (const auto &kv : feat_map)
  {
    const VOXEL_LOCATION &loc = kv.first;
    V3D voxel_center((loc.x + 0.5) * visual_voxel_size,
                     (loc.y + 0.5) * visual_voxel_size,
                     (loc.z + 0.5) * visual_voxel_size);
    ordered_voxels.emplace_back((voxel_center - center).squaredNorm(), loc);
  }

  std::sort(ordered_voxels.begin(), ordered_voxels.end(),
            [](const std::pair<double, VOXEL_LOCATION> &a, const std::pair<double, VOXEL_LOCATION> &b)
            {
              return a.first > b.first;
            });

  for (const auto &item : ordered_voxels)
  {
    const bool still_over_voxels = visual_map_max_voxels > 0 && feat_map.size() > static_cast<size_t>(visual_map_max_voxels);
    const bool still_over_points = visual_map_max_total_points > 0 && total_points > static_cast<size_t>(visual_map_max_total_points);
    if (!still_over_voxels && !still_over_points) break;

    auto it = feat_map.find(item.second);
    if (it == feat_map.end() || it->second == nullptr) continue;

    VOXEL_POINTS *voxel = it->second;
    const size_t voxel_points = voxel->voxel_points.size();
    delete voxel;
    feat_map.erase(it);
    if (total_points >= voxel_points) total_points -= voxel_points;
    else total_points = 0;
    ++removed_voxels;
    removed_points += voxel_points;
  }

  if (removed_points > 0 || removed_voxels > 0)
  {
    printf("[ VIO ] Prune visual map: remove %zu points, %zu voxels, remain voxels=%zu, remain points=%zu\n",
           removed_points, removed_voxels, feat_map.size(), total_points);
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
        
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;
  
  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  const int configured_cap = std::max(1, visual_map_max_add_per_frame);
  int max_add_this_frame = configured_cap;
  if (visual_map_max_total_points > 0)
  {
    const size_t total_points_now = getVisualPointCount();
    if (total_points_now >= static_cast<size_t>(visual_map_max_total_points))
    {
      printf("[ VIO ] Skip appending map points: visual map already at cap (%zu/%d)\n",
             total_points_now, visual_map_max_total_points);
      return;
    }
    const size_t remain = static_cast<size_t>(visual_map_max_total_points) - total_points_now;
    max_add_this_frame = std::min(max_add_this_frame, static_cast<int>(remain));
  }

  if (max_add_this_frame <= 0) return;

  const float min_corner_score = std::max(0.0f, visual_map_min_shi_tomasi_score);

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    const V3D &normal = pg[i].normal;
    const V3D &pt = pg[i].point_w;
    if (!std::isfinite(pt[0]) || !std::isfinite(pt[1]) || !std::isfinite(pt[2])) continue;
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) continue;
    if (normal.squaredNorm() < 1e-12) continue;

    V2D pc(new_frame_->w2c(pt));
    if (!std::isfinite(pc[0]) || !std::isfinite(pc[1])) continue;

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
      if (index < 0 || index >= length) continue;

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (!std::isfinite(cur_value)) continue;
        if (cur_value < min_corner_score) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    const V3D &pt = visual_submap->add_from_voxel_map[j].point_w;
    const V3D &normal = visual_submap->add_from_voxel_map[j].normal;
    if (!std::isfinite(pt[0]) || !std::isfinite(pt[1]) || !std::isfinite(pt[2])) continue;
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) continue;
    if (normal.squaredNorm() < 1e-12) continue;

    V2D pc(new_frame_->w2c(pt));
    if (!std::isfinite(pc[0]) || !std::isfinite(pc[1])) continue;

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
      if (index < 0 || index >= length) continue;

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (!std::isfinite(cur_value)) continue;
        if (cur_value < min_corner_score) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (add >= max_add_this_frame) break;

    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;
      if (!std::isfinite(pt[0]) || !std::isfinite(pt[1]) || !std::isfinite(pt[2])) continue;
      if (!std::isfinite(pt_var.normal[0]) || !std::isfinite(pt_var.normal[1]) || !std::isfinite(pt_var.normal[2])) continue;
      if (pt_var.normal.squaredNorm() < 1e-12) continue;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      if (norm_vec.squaredNorm() < 1e-12) continue;
      norm_vec.normalize();

      V3D dir(new_frame_->T_f_w_ * pt);
      if (!std::isfinite(dir[0]) || !std::isfinite(dir[1]) || !std::isfinite(dir[2])) continue;
      const double dir_norm = dir.norm();
      if (dir_norm < 1e-6) continue;
      dir /= dir_norm;

      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));
      if (!std::isfinite(pc[0]) || !std::isfinite(pc[1])) continue;
      if (!new_frame_->cam_->isInFrame(pc.cast<int>(), border)) continue;

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      if (insertPointIntoVoxelMap(pt_new))
      {
        add += 1;
      }
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points (cap=%d, score>=%.1f)\n",
         add, max_add_this_frame, min_corner_score);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;

    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      float *patch_temp = new float[patch_size_total];
      getImagePatch(img, pc, patch_temp, 0);
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    if (n_meas < min_update_meas)
    {
      (*state) = old_state;
      EKF_end = true;
      break;
    }

    error = error / n_meas;
    if (!std::isfinite(error))
    {
      (*state) = old_state;
      EKF_end = true;
      break;
    }

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
        MD(DIM_STATE, 1) solution =
          (-K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0)).eval();
        V3D rot_add = solution.block<3, 1>(0, 0);
        V3D t_add = solution.block<3, 1>(3, 0);
      const double rot_step_deg = rot_add.norm() * 57.3;
      const double trans_step_m = t_add.norm();
      const double max_rot_step_deg = std::max(0.1, max_state_update_rot_deg);
      const double max_trans_step_m = std::max(0.01, max_state_update_trans_m);
      double step_scale = 1.0;
      if (rot_step_deg > max_rot_step_deg) { step_scale = std::min(step_scale, max_rot_step_deg / std::max(rot_step_deg, 1e-6)); }
      if (trans_step_m > max_trans_step_m) { step_scale = std::min(step_scale, max_trans_step_m / std::max(trans_step_m, 1e-6)); }
      if (step_scale < 1.0)
      {
        solution *= step_scale;
        rot_add = solution.block<3, 1>(0, 0);
        t_add = solution.block<3, 1>(3, 0);
      }

      (*state) += solution;

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;
          
          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    if (n_meas < min_update_meas)
    {
      (*state) = old_state;
      EKF_end = true;
      break;
    }

    error = error / n_meas;
    if (!std::isfinite(error))
    {
      (*state) = old_state;
      EKF_end = true;
      break;
    }
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      V3D rot_add = solution.block<3, 1>(0, 0);
      V3D t_add = solution.block<3, 1>(3, 0);
      const double rot_step_deg = rot_add.norm() * 57.3;
      const double trans_step_m = t_add.norm();
      const double max_rot_step_deg = std::max(0.1, max_state_update_rot_deg);
      const double max_trans_step_m = std::max(0.01, max_state_update_trans_m);
      double step_scale = 1.0;
      if (rot_step_deg > max_rot_step_deg) { step_scale = std::min(step_scale, max_rot_step_deg / std::max(rot_step_deg, 1e-6)); }
      if (trans_step_m > max_trans_step_m) { step_scale = std::min(step_scale, max_trans_step_m / std::max(trans_step_m, 1e-6)); }
      if (step_scale < 1.0)
      {
        solution *= step_scale;
        rot_add = solution.block<3, 1>(0, 0);
        t_add = solution.block<3, 1>(3, 0);
      }

      (*state) += solution;

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

namespace
{
constexpr double kRadToDeg = 57.29577951308232;

cv::Point2f markerCenter2D(const std::vector<cv::Point2f> &corner_pts)
{
  cv::Point2f center(0.0f, 0.0f);
  for (const auto &pt : corner_pts) center += pt;
  center *= 0.25f;
  return center;
}

bool checkQuadGeometry2D(const std::vector<std::vector<cv::Point2f>> &corners,
                         double min_area_px,
                         double &quad_area_px)
{
  quad_area_px = 0.0;
  if (corners.size() != 4) return false;

  std::vector<cv::Point2f> centers;
  centers.reserve(4);
  for (const auto &corner_pts : corners)
  {
    if (corner_pts.size() != 4) return false;
    centers.push_back(markerCenter2D(corner_pts));
  }

  std::vector<cv::Point2f> hull;
  cv::convexHull(centers, hull);
  if (hull.size() != 4) return false;

  quad_area_px = std::fabs(cv::contourArea(hull));
  return quad_area_px >= min_area_px;
}

bool checkPairwiseDistanceConsistency(const std::vector<Eigen::Vector3d> &points,
                                      double dx,
                                      double dy,
                                      double rel_tol,
                                      double &max_rel_err)
{
  max_rel_err = 0.0;
  if (points.size() != 4) return false;

  const double expected_dx = 2.0 * std::fabs(dx);
  const double expected_dy = 2.0 * std::fabs(dy);
  const double expected_diag = 2.0 * std::sqrt(dx * dx + dy * dy);

  std::array<double, 6> expected = {
      expected_dx, expected_dx,
      expected_dy, expected_dy,
      expected_diag, expected_diag};
  std::sort(expected.begin(), expected.end());

  std::array<double, 6> measured;
  int idx = 0;
  for (size_t i = 0; i < points.size(); i++)
  {
    for (size_t j = i + 1; j < points.size(); j++)
    {
      measured[idx++] = (points[i] - points[j]).norm();
    }
  }
  std::sort(measured.begin(), measured.end());

  for (size_t i = 0; i < expected.size(); i++)
  {
    const double denom = std::max(expected[i], 1e-6);
    const double rel_err = std::fabs(measured[i] - expected[i]) / denom;
    max_rel_err = std::max(max_rel_err, rel_err);
    if (rel_err > rel_tol) return false;
  }

  return true;
}

bool checkNormalConsistency(const std::vector<VIOManager::ArucoObservation> &aruco_obs,
                           double max_normal_diff_deg,
                           double &observed_max_diff_deg)
{
  observed_max_diff_deg = 0.0;
  if (aruco_obs.size() != 4) return false;

  std::vector<Eigen::Vector3d> normals;
  normals.reserve(4);

  const Eigen::Vector3d ref_n = aruco_obs.front().R_cam_marker.col(2).normalized();
  normals.push_back(ref_n);

  for (size_t i = 1; i < aruco_obs.size(); i++)
  {
    Eigen::Vector3d n = aruco_obs[i].R_cam_marker.col(2).normalized();
    if (n.dot(ref_n) < 0.0) n = -n;
    normals.push_back(n);
  }

  Eigen::Vector3d avg_n = Eigen::Vector3d::Zero();
  for (const auto &n : normals) avg_n += n;
  if (avg_n.norm() < 1e-6) return false;
  avg_n.normalize();

  for (const auto &n : normals)
  {
    const double cos_angle = std::clamp(n.dot(avg_n), -1.0, 1.0);
    const double diff_deg = std::acos(cos_angle) * kRadToDeg;
    observed_max_diff_deg = std::max(observed_max_diff_deg, diff_deg);
  }

  return observed_max_diff_deg <= max_normal_diff_deg;
}

double computeCenterSpread(const std::vector<VIOManager::ArucoObservation> &aruco_obs,
                           const Eigen::Vector3d &board_center)
{
  if (aruco_obs.empty()) return 0.0;

  double sqr_sum = 0.0;
  for (const auto &obs : aruco_obs)
  {
    sqr_sum += (obs.tvec - board_center).squaredNorm();
  }
  return std::sqrt(sqr_sum / static_cast<double>(aruco_obs.size()));
}

double computeRotationDispersionDeg(const std::vector<VIOManager::ArucoObservation> &aruco_obs,
                                    const Eigen::Matrix3d &avg_rotation)
{
  if (aruco_obs.empty()) return 0.0;

  double max_diff_deg = 0.0;
  for (const auto &obs : aruco_obs)
  {
    const Eigen::Matrix3d delta_R = avg_rotation.transpose() * obs.R_cam_marker;
    Eigen::AngleAxisd aa(delta_R);
    const double diff_deg = std::fabs(aa.angle()) * kRadToDeg;
    max_diff_deg = std::max(max_diff_deg, diff_deg);
  }
  return max_diff_deg;
}

bool estimateBoardPoseFromCentersPnP(const std::vector<std::vector<cv::Point2f>> &corners,
                                     double half_dx,
                                     double half_dy,
                                     const cv::Mat &camera_matrix,
                                     const cv::Mat &dist_coeffs,
                                     Eigen::Vector3d &board_center,
                                     Eigen::Matrix3d &board_rotation,
                                     double &mean_reproj_error_px)
{
  mean_reproj_error_px = std::numeric_limits<double>::infinity();
  if (corners.size() != 4) return false;

  std::vector<cv::Point2f> image_centers;
  image_centers.reserve(4);
  for (const auto &corner_pts : corners)
  {
    if (corner_pts.size() != 4) return false;
    image_centers.push_back(markerCenter2D(corner_pts));
  }

  const std::vector<cv::Point3f> object_points = {
      cv::Point3f(static_cast<float>(-half_dx), static_cast<float>(half_dy), 0.0f),
      cv::Point3f(static_cast<float>(half_dx), static_cast<float>(half_dy), 0.0f),
      cv::Point3f(static_cast<float>(-half_dx), static_cast<float>(-half_dy), 0.0f),
      cv::Point3f(static_cast<float>(half_dx), static_cast<float>(-half_dy), 0.0f)};

  std::array<int, 4> perm = {0, 1, 2, 3};
  cv::Vec3d best_rvec(0, 0, 0);
  cv::Vec3d best_tvec(0, 0, 0);
  bool found = false;

  do
  {
    std::vector<cv::Point2f> ordered_image_points = {
        image_centers[perm[0]], image_centers[perm[1]], image_centers[perm[2]], image_centers[perm[3]]};

    cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);
    const bool pnp_ok = cv::solvePnP(object_points,
                                     ordered_image_points,
                                     camera_matrix,
                                     dist_coeffs,
                                     rvec,
                                     tvec,
                                     false,
                                     cv::SOLVEPNP_ITERATIVE);
    if (!pnp_ok) continue;

    cv::solvePnPRefineLM(object_points, ordered_image_points, camera_matrix, dist_coeffs, rvec, tvec);

    std::vector<cv::Point2f> reproj_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, reproj_points);

    double reproj_err_sum = 0.0;
    for (size_t i = 0; i < reproj_points.size(); i++)
    {
      reproj_err_sum += cv::norm(reproj_points[i] - ordered_image_points[i]);
    }
    const double mean_err = reproj_err_sum / static_cast<double>(reproj_points.size());

    if (!found || mean_err < mean_reproj_error_px)
    {
      found = true;
      mean_reproj_error_px = mean_err;
      best_rvec = rvec;
      best_tvec = tvec;
    }
  } while (std::next_permutation(perm.begin(), perm.end()));

  if (!found) return false;

  cv::Mat R_cv;
  cv::Rodrigues(best_rvec, R_cv);
  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      board_rotation(r, c) = R_cv.at<double>(r, c);
    }
  }
  board_center = Eigen::Vector3d(best_tvec[0], best_tvec[1], best_tvec[2]);
  return true;
}
} // namespace

// 添加辅助函数：反对称矩阵
Eigen::Matrix3d VIOManager::skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// 添加辅助函数：指数映射
Eigen::Matrix3d VIOManager::Exp(const Eigen::Vector3d& w)
{
    double theta = w.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d W = skewSymmetric(w / theta);
    return Eigen::Matrix3d::Identity() + sin(theta) * W + (1 - cos(theta)) * W * W;
}

void VIOManager::detect_qr(cv::Mat img)
{
  const double t_aruco_begin = omp_get_wtime();
  aruco_time_detect_markers = 0.0;
  aruco_time_draw = 0.0;
  aruco_time_group_gate = 0.0;
  aruco_time_pose_estimate = 0.0;
  aruco_time_pnp = 0.0;
  aruco_board_candidates = 0;
  aruco_board_accepted = 0;

  current_board_observations_.clear();

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;

  const double t_detect_begin = omp_get_wtime();
  cv::aruco::detectMarkers(img, dictionary_, corners, ids, parameters_, rejectedCandidates);
  aruco_time_detect_markers = omp_get_wtime() - t_detect_begin;
  
  if (ids.empty()) 
  {
    aruco_time_total = omp_get_wtime() - t_aruco_begin;
    return;
  }

  const double t_draw_begin = omp_get_wtime();
  draw_qr(ids, corners, rejectedCandidates);
  aruco_time_draw = omp_get_wtime() - t_draw_begin;

  const double t_group_begin = omp_get_wtime();
  std::map<int, std::vector<size_t>> grouped_indices;
  for (size_t i = 0; i < ids.size(); i++)
  {
    grouped_indices[ids[i]].push_back(i);
  }
  aruco_board_candidates = static_cast<int>(grouped_indices.size());

  for (const auto& group : grouped_indices)
  {
    double t_gate_anchor = omp_get_wtime();
    const int board_id = group.first;
    const std::vector<size_t>& indices = group.second;

    if (indices.size() != 4)
    {
      printf("\033[1;33m[Aruco] Skip board %d: require exactly 4 same-ID markers, got %zu.\033[0m\n",
             board_id, indices.size());
      continue;
    }

    std::vector<std::vector<cv::Point2f>> board_corners;
    board_corners.reserve(4);
    for (size_t idx : indices)
    {
      board_corners.push_back(corners[idx]);
    }

    double quad_area_px = 0.0;
    if (!checkQuadGeometry2D(board_corners, aruco_min_quad_area_px, quad_area_px))
    {
      printf("\033[1;33m[Aruco] Skip board %d: invalid 2D quad geometry, area %.1f px^2.\033[0m\n",
             board_id, quad_area_px);
      continue;
    }

    aruco_time_group_gate += omp_get_wtime() - t_gate_anchor;

    const double t_pose_begin = omp_get_wtime();
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(board_corners, marker_size, cameraMatrix_, distCoeffs_, rvecs, tvecs);
    aruco_time_pose_estimate += omp_get_wtime() - t_pose_begin;
    if (rvecs.size() != 4 || tvecs.size() != 4)
    {
      printf("\033[1;33m[Aruco] Skip board %d: pose estimation size mismatch.\033[0m\n", board_id);
      continue;
    }

    t_gate_anchor = omp_get_wtime();

    auto it = board_world_positions_.find(board_id);
    if (it == board_world_positions_.end())
    {
      board_world_positions_[board_id] = Eigen::Vector3d::Zero();
      board_world_orientations_[board_id] = Eigen::Matrix3d::Identity();
      board_world_flag_[board_id] = false;
      printf("\033[1;33m[Aruco] Auto-register board %d as uninitialized.\033[0m\n", board_id);
    }

    std::vector<ArucoObservation> aruco_obs;
    aruco_obs.reserve(4);
    std::vector<Eigen::Vector3d> marker_positions;
    marker_positions.reserve(4);

    double min_marker_z = std::numeric_limits<double>::max();
    double max_marker_z = -std::numeric_limits<double>::max();
    bool depth_valid = true;

    for (size_t i = 0; i < 4; i++)
    {
      ArucoObservation obs;
      obs.id = board_id;
      obs.tvec = Eigen::Vector3d(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

      const double marker_distance = obs.tvec.norm();
      if (marker_distance < aruco_min_marker_depth || marker_distance > aruco_max_marker_depth)
      {
        printf("\033[1;33m[Aruco] Skip board %d: marker distance %.3f m out of range.\033[0m\n",
               board_id, marker_distance);
        depth_valid = false;
        break;
      }

      min_marker_z = std::min(min_marker_z, obs.tvec.z());
      max_marker_z = std::max(max_marker_z, obs.tvec.z());

      cv::Mat R_cam_marker_cv;
      cv::Rodrigues(rvecs[i], R_cam_marker_cv);
      for (int row = 0; row < 3; row++)
      {
        for (int col = 0; col < 3; col++)
        {
          obs.R_cam_marker(row, col) = R_cam_marker_cv.at<double>(row, col);
        }
      }

      aruco_obs.push_back(obs);
      marker_positions.push_back(obs.tvec);

      cv::drawFrameAxes(img_cp, cameraMatrix_, distCoeffs_, rvecs[i], tvecs[i], 0.1);
    }
    if (!depth_valid) continue;

    if ((max_marker_z - min_marker_z) > aruco_max_marker_depth_diff)
    {
      printf("\033[1;33m[Aruco] Skip board %d: depth spread %.3f m too large.\033[0m\n",
             board_id, max_marker_z - min_marker_z);
      continue;
    }

    double max_rel_pair_err = 0.0;
    if (!checkPairwiseDistanceConsistency(marker_positions,
                                          board_config_.delta_width_qr_center,
                                          board_config_.delta_height_qr_center,
                                          aruco_pair_distance_rel_tol,
                                          max_rel_pair_err))
    {
      printf("\033[1;33m[Aruco] Skip board %d: pairwise distance inconsistency %.3f.\033[0m\n",
             board_id, max_rel_pair_err);
      continue;
    }

    double max_normal_diff_deg = 0.0;
    if (!checkNormalConsistency(aruco_obs, aruco_max_normal_diff_deg, max_normal_diff_deg))
    {
      printf("\033[1;33m[Aruco] Skip board %d: normal inconsistency %.2f deg.\033[0m\n",
             board_id, max_normal_diff_deg);
      continue;
    }

    aruco_time_group_gate += omp_get_wtime() - t_gate_anchor;

    Eigen::Vector3d board_center;
    Eigen::Matrix3d board_rotation;
    int valid_count = 4;
    double board_reproj_error_px = 0.0;
    const double t_pnp_begin = omp_get_wtime();
    const bool pnp_ok = estimateBoardPoseFromCentersPnP(board_corners,
                                                        board_config_.delta_width_qr_center,
                                                        board_config_.delta_height_qr_center,
                                                        cameraMatrix_,
                                                        distCoeffs_,
                                                        board_center,
                                                        board_rotation,
                                                        board_reproj_error_px);
    aruco_time_pnp += omp_get_wtime() - t_pnp_begin;
    if (!pnp_ok)
    {
      printf("\033[1;33m[Aruco] Skip board %d: joint board pose optimization failed.\033[0m\n", board_id);
      continue;
    }

    BoardObservation board_obs;
    board_obs.board_id = board_id;
    board_obs.center_tvec = board_center;
    board_obs.center_R_cam_board = board_rotation;
    board_obs.valid_count = valid_count;
    board_obs.geometry_valid = true;
    board_obs.center_spread_m = computeCenterSpread(aruco_obs, board_center);
    board_obs.rotation_dispersion_deg = computeRotationDispersionDeg(aruco_obs, board_rotation);
    current_board_observations_.push_back(board_obs);
    aruco_board_accepted++;

    printf("\033[1;32m[Aruco] Board %d accepted: center [%.3f, %.3f, %.3f], spread %.3f m, rot-disp %.2f deg, area %.1f px^2, reproj %.3f px\033[0m\n",
           board_id,
           board_center.x(),
           board_center.y(),
           board_center.z(),
           board_obs.center_spread_m,
           board_obs.rotation_dispersion_deg,
           quad_area_px,
           board_reproj_error_px);
  }

  aruco_time_group_gate = omp_get_wtime() - t_group_begin;
  aruco_time_total = omp_get_wtime() - t_aruco_begin;
}

void VIOManager::draw_qr(
  std::vector<int>& ids, 
  std::vector<std::vector<cv::Point2f>>& corners, 
  std::vector<std::vector<cv::Point2f>>& rejectedCandidates)
{
  // 绘制检测到的标签
  //if (ids.size() > 0) cv::aruco::drawDetectedMarkers(imageCopy_, corners, ids);

  // 遍历每一个检测到的 Aruco 标记
  for (size_t i = 0; i < ids.size(); i++)
  {
      // 获取当前 marker 的四个角点 [4个点, 2坐标(x,y)]，数据类型是 float
      std::vector<cv::Point2f> cornerPts = corners[i];

      // 将 float 型的角点转换为 int（OpenCV 绘图函数一般用 int）
      cv::Point2i pt1 = cv::Point2i(static_cast<int>(cornerPts[0].x), static_cast<int>(cornerPts[0].y));
      cv::Point2i pt2 = cv::Point2i(static_cast<int>(cornerPts[1].x), static_cast<int>(cornerPts[1].y));
      cv::Point2i pt3 = cv::Point2i(static_cast<int>(cornerPts[2].x), static_cast<int>(cornerPts[2].y));
      cv::Point2i pt4 = cv::Point2i(static_cast<int>(cornerPts[3].x), static_cast<int>(cornerPts[3].y));

      // 绘制 Aruco 方框的 4 条边 —— 使用 绿色 (BGR: 0, 255, 0)
      cv::line(img_cp, pt1, pt2, cv::Scalar(0, 255, 0), 2); // 上边
      cv::line(img_cp, pt2, pt3, cv::Scalar(0, 255, 0), 2); // 右边
      cv::line(img_cp, pt3, pt4, cv::Scalar(0, 255, 0), 2); // 下边
      cv::line(img_cp, pt4, pt1, cv::Scalar(0, 255, 0), 2); // 左边

      // 计算中心点
      cv::Point2f center(0, 0);
      for (const auto& p : cornerPts)
          center += p;
      center *= (1.0 / 4.0);  // 计算中心点

      int id = ids[i];
      
      // 修改文本内容和位置
      std::string label = "id = " + std::to_string(id);
      
      // 计算文本大小，用于精确定位
      int baseline = 0;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
      
      // 计算文本位置：中心靠右（向右偏移文本宽度的一半）
      cv::Point2i textPos(static_cast<int>(center.x) + textSize.width/4, 
                        static_cast<int>(center.y));
      
      cv::putText(img_cp, 
                label,  // 修改为 "id=数字" 格式
                textPos, 
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(0, 255, 0),  // 绿色
                2);
  }

  // 绘制被拒绝的候选区域（红色）
  if (!rejectedCandidates.empty()) 
  {
      for (const auto& candidate : rejectedCandidates) {
          // 将点转换为整数坐标
          std::vector<cv::Point> intPoints;
          for (const auto& point : candidate) {
              intPoints.push_back(cv::Point(static_cast<int>(point.x), 
                                        static_cast<int>(point.y)));
          }
          
          // 绘制多边形轮廓（红色，线宽2像素）
          cv::polylines(img_cp, intPoints, true, cv::Scalar(0, 0, 255), 2);
          
          // 可选：绘制角点（蓝色圆点）
          for (const auto& point : candidate) {
              cv::circle(img_cp, point, 5, cv::Scalar(255, 0, 0), -1);
          }
      }
  }

}

void VIOManager::updateStateWithBoardObservation()
{
  if (current_board_observations_.empty()) return;

  for (const auto& board_obs : current_board_observations_)
  {
    if (board_obs.valid_count != 4 || !board_obs.geometry_valid)
    {
      printf("\033[1;33m[Aruco] Skip update: invalid board observation (count=%d, geometry=%d).\033[0m\n",
             board_obs.valid_count, static_cast<int>(board_obs.geometry_valid));
      continue;
    }

    int board_id = board_obs.board_id;
    auto flag_it = board_world_flag_.find(board_id);
    if (flag_it == board_world_flag_.end())
    {
      board_world_flag_[board_id] = false;
      board_world_positions_[board_id] = Eigen::Vector3d::Zero();
      board_world_orientations_[board_id] = Eigen::Matrix3d::Identity();
      flag_it = board_world_flag_.find(board_id);
    }

    M3D R_wi(state->rot_end);
    V3D P_wi(state->pos_end);

    M3D R_ic = Rci.transpose();
    V3D P_ic = -R_ic * Pci;

    M3D R_wc = R_wi * R_ic;
    V3D P_wc = P_wi + R_wi * P_ic;

    V3D P_w_board_estimated = R_wc * board_obs.center_tvec + P_wc;
    M3D R_w_board_estimated = R_wc * board_obs.center_R_cam_board;

    if (!flag_it->second)
    {
      printf("\033[1;33m===========================================\033[0m\n");
      printf("\033[1;33m[Board %d] FIRST OBSERVATION - Initializing Landmark\033[0m\n", board_id);

      board_world_positions_[board_id] = P_w_board_estimated;
      board_world_orientations_[board_id] = R_w_board_estimated;
      board_world_flag_[board_id] = true;

      printf("\033[1;33m  Initialized world position: [%.3f, %.3f, %.3f] meters\033[0m\n",
            P_w_board_estimated.x(), P_w_board_estimated.y(), P_w_board_estimated.z());

      Eigen::Vector3d euler_angles = R_w_board_estimated.eulerAngles(0, 1, 2);
      printf("\033[1;33m  Initialized orientation (RPY): [%.1f, %.1f, %.1f] degrees\033[0m\n",
            euler_angles.x() * 180.0 / M_PI,
            euler_angles.y() * 180.0 / M_PI,
            euler_angles.z() * 180.0 / M_PI);

      printf("\033[1;33m  Valid markers: %d\033[0m\n", board_obs.valid_count);
      printf("\033[1;33m===========================================\033[0m\n");
      continue;
    }

    const Eigen::Vector3d& P_w_board = board_world_positions_[board_id];
    const Eigen::Matrix3d& R_w_board = board_world_orientations_[board_id];

    printf("\033[1;32m===========================================\033[0m\n");
    printf("\033[1;32m[Board %d] RE-OBSERVATION - Updating State\033[0m\n", board_id);
    printf("\033[1;32m  Estimated:  [%.3f, %.3f, %.3f] meters\033[0m\n",
          P_w_board_estimated.x(), P_w_board_estimated.y(), P_w_board_estimated.z());
    printf("\033[1;32m  Landmark:   [%.3f, %.3f, %.3f] meters\033[0m\n",
          P_w_board.x(), P_w_board.y(), P_w_board.z());

    V3D position_error = P_w_board_estimated - P_w_board;
    const double position_error_norm = position_error.norm();
    printf("\033[1;32m  Position Error: [%.3f, %.3f, %.3f] meters, Norm: %.3fm\033[0m\n",
        position_error.x(), position_error.y(), position_error.z(), position_error_norm);

    const M3D R_flip_pi_x = (Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const M3D R_flip_pi_y = (Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY())).toRotationMatrix();
    const M3D R_flip_pi_z = (Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ())).toRotationMatrix();

    auto computeRotationErrorDeg = [](const M3D &R_candidate, const M3D &R_ref) {
      Eigen::AngleAxisd aa(R_candidate * R_ref.transpose());
      return aa.angle() * 180.0 / M_PI;
    };

    const std::array<M3D, 4> candidate_flips = {
        M3D::Identity(),
        R_flip_pi_x,
        R_flip_pi_y,
        R_flip_pi_z};

    M3D R_cam_board_used = board_obs.center_R_cam_board;
    double orientation_error_deg = std::numeric_limits<double>::infinity();
    double orientation_error_before = computeRotationErrorDeg(R_w_board_estimated, R_w_board);

    for (const auto &R_flip : candidate_flips)
    {
      const M3D R_cam_candidate = board_obs.center_R_cam_board * R_flip;
      const M3D R_w_candidate = R_wc * R_cam_candidate;
      const double err_deg = computeRotationErrorDeg(R_w_candidate, R_w_board);
      if (err_deg < orientation_error_deg)
      {
        orientation_error_deg = err_deg;
        R_cam_board_used = R_cam_candidate;
        R_w_board_estimated = R_w_candidate;
      }
    }

    if (orientation_error_deg + 1e-3 < orientation_error_before)
    {
      printf("\033[1;33m[Aruco] Board %d: resolve orientation ambiguity (%.2f -> %.2f deg).\033[0m\n",
             board_id,
             orientation_error_before,
             orientation_error_deg);
    }

    printf("\033[1;32m  Orientation Error: %.2f degrees\033[0m\n", orientation_error_deg);

    const Eigen::Vector3d n_est = R_w_board_estimated.col(2).normalized();
    const Eigen::Vector3d n_ref = R_w_board.col(2).normalized();
    const double normal_error_deg =
        std::acos(std::clamp(n_est.dot(n_ref), -1.0, 1.0)) * 180.0 / M_PI;
    printf("\033[1;32m  Normal Error: %.2f degrees\033[0m\n", normal_error_deg);

    if (position_error_norm > aruco_max_position_residual ||
        normal_error_deg > aruco_normal_gate_deg)
    {
      printf("\033[1;33m[Aruco] Reject update for board %d: residual gate failed (pos %.3f/%.3f m, normal %.2f/%.2f deg).\033[0m\n",
             board_id,
             position_error_norm,
             aruco_max_position_residual,
             normal_error_deg,
             aruco_normal_gate_deg);
      continue;
    }

    double distance = board_obs.center_tvec.norm();
    printf("\033[1;32m  Distance from camera: %.3f meters\033[0m\n", distance);

    Eigen::Vector3d euler_angles = R_w_board.eulerAngles(0, 1, 2);
    printf("\033[1;32m  Landmark orientation (RPY): [%.1f, %.1f, %.1f] degrees\033[0m\n",
          euler_angles.x() * 180.0 / M_PI,
          euler_angles.y() * 180.0 / M_PI,
          euler_angles.z() * 180.0 / M_PI);

    printf("\033[1;32m  Valid markers: %d\033[0m\n", board_obs.valid_count);

    int observed_count = 0;
    for (const auto& flag : board_world_flag_)
    {
      if (flag.second) observed_count++;
    }
    printf("\033[1;32m  Total observed landmarks: %d/%zu\033[0m\n", observed_count, board_world_flag_.size());
    printf("\033[1;32m===========================================\033[0m\n");

    V3D position_residual = P_w_board - P_w_board_estimated;

    const bool use_orientation_update =
        aruco_use_orientation_update && (orientation_error_deg <= aruco_max_orientation_residual_deg);

    Eigen::MatrixXd H_aruco;
    Eigen::MatrixXd R_aruco;
    Eigen::VectorXd z_aruco;

    M3D p_hat = skewSymmetric(board_obs.center_tvec);
    MD(3, 3) J_pos_R = -R_wc * p_hat;
    MD(3, 3) J_pos_t = -M3D::Identity();

    double quality_weight = 1.0 + board_obs.center_spread_m + board_obs.rotation_dispersion_deg / 45.0;
    quality_weight = std::max(1.0, quality_weight);

    if (use_orientation_update)
    {
      M3D R_cw = R_wc.transpose();
      M3D residual_R = R_cam_board_used.transpose() * R_cw * R_w_board;
      Eigen::AngleAxisd angle_axis(residual_R);
      V3D orientation_residual = angle_axis.angle() * angle_axis.axis();

      z_aruco.resize(6);
      z_aruco.segment<3>(0) = position_residual;
      z_aruco.segment<3>(3) = orientation_residual;

      H_aruco = Eigen::MatrixXd::Zero(6, 6);
      MD(3, 3) J_ori_R = -M3D::Identity();
      MD(3, 3) J_ori_t = MD(3, 3)::Zero();
      H_aruco.block<3, 3>(0, 0) = J_pos_R;
      H_aruco.block<3, 3>(0, 3) = J_pos_t;
      H_aruco.block<3, 3>(3, 0) = J_ori_R;
      H_aruco.block<3, 3>(3, 3) = J_ori_t;

      R_aruco = Eigen::MatrixXd::Zero(6, 6);
      R_aruco.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * aruco_position_noise_base * quality_weight;
      R_aruco.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * aruco_orientation_noise_base * quality_weight;
    }
    else
    {
      z_aruco = position_residual;
      H_aruco = Eigen::MatrixXd::Zero(3, 6);
      H_aruco.block<3, 3>(0, 0) = J_pos_R;
      H_aruco.block<3, 3>(0, 3) = J_pos_t;
      R_aruco = Eigen::MatrixXd::Identity(3, 3) * aruco_position_noise_base * quality_weight;
    }

    Eigen::MatrixXd H_T = H_aruco.transpose();
    Eigen::MatrixXd S = H_aruco * state->cov.block<6, 6>(0, 0) * H_T + R_aruco;
    Eigen::MatrixXd K = state->cov.block<6, 6>(0, 0) * H_T * S.inverse();

    Eigen::VectorXd dx = K * z_aruco;

    V3D rot_add = dx.head<3>();
    V3D trans_add = dx.tail<3>();
    const double rot_step_deg = rot_add.norm() * 57.3;
    const double trans_step_m = trans_add.norm();
    const double max_rot_step_deg = std::max(0.1, aruco_update_max_rot_step_deg);
    const double max_trans_step_m = std::max(0.01, aruco_update_max_trans_step_m);
    double step_scale = 1.0;
    if (rot_step_deg > max_rot_step_deg) step_scale = std::min(step_scale, max_rot_step_deg / std::max(rot_step_deg, 1e-6));
    if (trans_step_m > max_trans_step_m) step_scale = std::min(step_scale, max_trans_step_m / std::max(trans_step_m, 1e-6));
    if (step_scale < 1.0) dx *= step_scale;

    state->rot_end = state->rot_end * Exp(dx.head<3>());
    state->pos_end += dx.tail<3>();

    Eigen::MatrixXd I_KH = Eigen::Matrix<double, 6, 6>::Identity() - K * H_aruco;
    state->cov.block<6, 6>(0, 0) = I_KH * state->cov.block<6, 6>(0, 0) * I_KH.transpose() + K * R_aruco * K.transpose();

        printf("[Aruco] Updated with board %d, mode=%s, residual norm: %.6f, quality weight: %.3f\n",
          board_id,
          use_orientation_update ? "pos+ori" : "pos-only",
          z_aruco.norm(),
          quality_weight);
  }
}

void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  (void)img_time;

  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);
  const double t_retrieve = omp_get_wtime() - t1;
  const bool run_aruco_this_frame = aruco_landmarks_en && (aruco_process_stride <= 1 || (frame_count % aruco_process_stride == 0));
  if (run_aruco_this_frame)
  {
    detect_qr(img);
  }
  else
  {
    current_board_observations_.clear();
    aruco_time_detect_markers = 0.0;
    aruco_time_draw = 0.0;
    aruco_time_group_gate = 0.0;
    aruco_time_pose_estimate = 0.0;
    aruco_time_pnp = 0.0;
    aruco_time_total = 0.0;
    aruco_board_candidates = 0;
    aruco_board_accepted = 0;
  }

  const bool low_track_force_update =
      low_track_force_update_stride > 0 &&
      total_points >= low_track_force_min_points &&
      (frame_count % low_track_force_update_stride == 0);
  const bool skip_visual_ekf = (total_points < min_retrieve_points) && !low_track_force_update;
  double t2 = omp_get_wtime();

  if (!skip_visual_ekf)
  {
    if (low_track_force_update && total_points < min_retrieve_points)
    {
      printf("[ VIO ] Force EKF update under low tracks (%d < %d), stride=%d, min_force_pts=%d\n",
             total_points, min_retrieve_points, low_track_force_update_stride, low_track_force_min_points);
    }

    computeJacobianAndUpdateEKF(img);
    if (run_aruco_this_frame)
    {
      const double t_aruco_update_begin = omp_get_wtime();
      updateStateWithBoardObservation();
      aruco_time_update = omp_get_wtime() - t_aruco_update_begin;
    }
    else
    {
      aruco_time_update = 0.0;
    }
  }
  else
  {
    const bool aruco_update_ran = run_aruco_this_frame && !current_board_observations_.empty();

    if (aruco_update_ran)
    {
      const double t_aruco_update_begin = omp_get_wtime();
      updateStateWithBoardObservation();
      aruco_time_update = omp_get_wtime() - t_aruco_update_begin;
    }
    else
    {
      aruco_time_update = 0.0;
    }

    const double t_map_gen_begin = omp_get_wtime();
    generateVisualMapPoints(img, pg);
    pruneVisualMap();
    const double t_map_gen = omp_get_wtime() - t_map_gen_begin;

    const double t_low_exit = omp_get_wtime();
    const double vio_time_now = t_low_exit - t1;
    frame_count++;
    ave_total = ave_total * (frame_count - 1) / frame_count + vio_time_now / frame_count;

    printf("[ VIO ] Skip visual EKF update: low tracked points (%d < %d), map-gen=%.6f s, total=%.6f s.\n",
           total_points, min_retrieve_points, t_map_gen, vio_time_now);

    printf("[ VIO Time ] skip-path: retrieve=%.6f, map_gen=%.6f, aruco_detect=%.6f, aruco_update=%.6f, total=%.6f, avg=%.6f\n",
          t_retrieve,
           t_map_gen,
           aruco_time_total,
           aruco_time_update,
           vio_time_now,
           ave_total);

    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                    VIO Time (Skip Path)                     |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t_retrieve);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t_map_gen);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", vio_time_now);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

    if (run_aruco_this_frame)
    {
      aruco_profile_frames++;
      aruco_ave_time_total = aruco_ave_time_total * (aruco_profile_frames - 1) / aruco_profile_frames +
                             (aruco_time_total + aruco_time_update) / aruco_profile_frames;

      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
      printf("\033[1;35m|                    Aruco Time (Skip Path)                   |\033[0m\n");
      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
      printf("\033[1;35m| %-29s | %-27d |\033[0m\n", "Board Candidates", aruco_board_candidates);
      printf("\033[1;35m| %-29s | %-27d |\033[0m\n", "Board Accepted", aruco_board_accepted);
      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
      printf("\033[1;35m| %-29s | %-27s |\033[0m\n", "Aruco Stage", "Time (secs)");
      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "detectMarkers", aruco_time_detect_markers);
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "draw_qr", aruco_time_draw);
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "group+geometry gates", aruco_time_group_gate);
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "estimatePoseSingleMarkers", aruco_time_pose_estimate);
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "board pose PnP", aruco_time_pnp);
      if (aruco_update_ran)
      {
        printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "state update", aruco_time_update);
      }
      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Current Aruco Total", aruco_time_total + aruco_time_update);
      printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Average Aruco Total", aruco_ave_time_total);
      printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    }

    {
      vector<string> lines;
      {
        std::ostringstream oss;
        oss << "[ VIO ] Skip visual EKF update: low tracked points (" << total_points << " < "
            << min_retrieve_points << "), map-gen=" << formatDouble6(t_map_gen)
            << " s, total=" << formatDouble6(vio_time_now) << " s.";
        lines.push_back(oss.str());
      }
      {
        std::ostringstream oss;
        oss << "[ VIO Time ] skip-path: retrieve=" << formatDouble6(t_retrieve)
            << ", map_gen=" << formatDouble6(t_map_gen)
            << ", aruco_detect=" << formatDouble6(aruco_time_total)
            << ", aruco_update=" << formatDouble6(aruco_time_update)
            << ", total=" << formatDouble6(vio_time_now)
            << ", avg=" << formatDouble6(ave_total);
        lines.push_back(oss.str());
      }
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back("|                    VIO Time (Skip Path)                     |");
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Sparse Map Size", std::to_string(feat_map.size())));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Algorithm Stage", "Time (secs)"));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("retrieveFromVisualSparseMap", formatDouble6(t_retrieve)));
      lines.push_back(makeTableRow("generateVisualMapPoints", formatDouble6(t_map_gen)));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Current Total Time", formatDouble6(vio_time_now)));
      lines.push_back(makeTableRow("Average Total Time", formatDouble6(ave_total)));
      lines.push_back("+-------------------------------------------------------------+");

      if (run_aruco_this_frame)
      {
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back("|                    Aruco Time (Skip Path)                   |");
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back(makeTableRow("Board Candidates", std::to_string(aruco_board_candidates)));
        lines.push_back(makeTableRow("Board Accepted", std::to_string(aruco_board_accepted)));
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back(makeTableRow("Aruco Stage", "Time (secs)"));
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back(makeTableRow("detectMarkers", formatDouble6(aruco_time_detect_markers)));
        lines.push_back(makeTableRow("draw_qr", formatDouble6(aruco_time_draw)));
        lines.push_back(makeTableRow("group+geometry gates", formatDouble6(aruco_time_group_gate)));
        lines.push_back(makeTableRow("estimatePoseSingleMarkers", formatDouble6(aruco_time_pose_estimate)));
        lines.push_back(makeTableRow("board pose PnP", formatDouble6(aruco_time_pnp)));
        if (aruco_update_ran)
        {
          lines.push_back(makeTableRow("state update", formatDouble6(aruco_time_update)));
        }
        lines.push_back("+-------------------------------------------------------------+");
        lines.push_back(makeTableRow("Current Aruco Total", formatDouble6(aruco_time_total + aruco_time_update)));
        lines.push_back(makeTableRow("Average Aruco Total", formatDouble6(aruco_ave_time_total)));
        lines.push_back("+-------------------------------------------------------------+");
      }

      appendTimingLogLines(lines);
    }

    return;
  }

  // Keep an explicit guard to avoid accidental fall-through if skip-branch
  // return is modified during experiments.
  if (skip_visual_ekf) return;

  double t3 = omp_get_wtime();

  generateVisualMapPoints(img, pg);
  
  double t4 = omp_get_wtime();
  
  plotTrackedPoints();

  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  pruneVisualMap();

  double t7 = omp_get_wtime();
  
  if(colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t_retrieve);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  if (run_aruco_this_frame)
  {
    aruco_profile_frames++;
    aruco_ave_time_total = aruco_ave_time_total * (aruco_profile_frames - 1) / aruco_profile_frames +
                           (aruco_time_total + aruco_time_update) / aruco_profile_frames;

    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;35m|                        Aruco Time                           |\033[0m\n");
    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;35m| %-29s | %-27d |\033[0m\n", "Board Candidates", aruco_board_candidates);
    printf("\033[1;35m| %-29s | %-27d |\033[0m\n", "Board Accepted", aruco_board_accepted);
    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;35m| %-29s | %-27s |\033[0m\n", "Aruco Stage", "Time (secs)");
    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "detectMarkers", aruco_time_detect_markers);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "draw_qr", aruco_time_draw);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "group+geometry gates", aruco_time_group_gate);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "estimatePoseSingleMarkers", aruco_time_pose_estimate);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "board pose PnP", aruco_time_pnp);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "state update", aruco_time_update);
    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Current Aruco Total", aruco_time_total + aruco_time_update);
    printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Average Aruco Total", aruco_ave_time_total);
    printf("\033[1;35m+-------------------------------------------------------------+\033[0m\n");
  }

  {
    vector<string> lines;
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back("|                         VIO Time                            |");
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("Sparse Map Size", std::to_string(feat_map.size())));
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("Algorithm Stage", "Time (secs)"));
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("retrieveFromVisualSparseMap", formatDouble6(t_retrieve)));
    lines.push_back(makeTableRow("computeJacobianAndUpdateEKF", formatDouble6(t3 - t2)));
    lines.push_back(makeTableRow("-> computeJacobian", formatDouble6(compute_jacobian_time)));
    lines.push_back(makeTableRow("-> updateEKF", formatDouble6(update_ekf_time)));
    lines.push_back(makeTableRow("generateVisualMapPoints", formatDouble6(t4 - t3)));
    lines.push_back(makeTableRow("updateVisualMapPoints", formatDouble6(t6 - t5)));
    lines.push_back(makeTableRow("updateReferencePatch", formatDouble6(t7 - t6)));
    lines.push_back("+-------------------------------------------------------------+");
    lines.push_back(makeTableRow("Current Total Time", formatDouble6(t7 - t1 - (t5 - t4))));
    lines.push_back(makeTableRow("Average Total Time", formatDouble6(ave_total)));
    lines.push_back("+-------------------------------------------------------------+");

    if (run_aruco_this_frame)
    {
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back("|                        Aruco Time                           |");
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Board Candidates", std::to_string(aruco_board_candidates)));
      lines.push_back(makeTableRow("Board Accepted", std::to_string(aruco_board_accepted)));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Aruco Stage", "Time (secs)"));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("detectMarkers", formatDouble6(aruco_time_detect_markers)));
      lines.push_back(makeTableRow("draw_qr", formatDouble6(aruco_time_draw)));
      lines.push_back(makeTableRow("group+geometry gates", formatDouble6(aruco_time_group_gate)));
      lines.push_back(makeTableRow("estimatePoseSingleMarkers", formatDouble6(aruco_time_pose_estimate)));
      lines.push_back(makeTableRow("board pose PnP", formatDouble6(aruco_time_pnp)));
      lines.push_back(makeTableRow("state update", formatDouble6(aruco_time_update)));
      lines.push_back("+-------------------------------------------------------------+");
      lines.push_back(makeTableRow("Current Aruco Total", formatDouble6(aruco_time_total + aruco_time_update)));
      lines.push_back(makeTableRow("Average Aruco Total", formatDouble6(aruco_ave_time_total)));
      lines.push_back("+-------------------------------------------------------------+");
    }

    appendTimingLogLines(lines);
  }
}
