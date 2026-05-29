/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;

  double fx, fy, cx, cy;
  double d0, d1, d2, d3;
  cv::Mat cameraMatrix_;
  cv::Mat distCoeffs_;
  cv::Ptr<cv::aruco::DetectorParameters> parameters_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  double marker_size;
  bool aruco_landmarks_en = false;
  double aruco_min_quad_area_px = 300.0;
  double aruco_pair_distance_rel_tol = 0.2;
  double aruco_max_normal_diff_deg = 15.0;
  double aruco_max_marker_depth_diff = 0.6;
  double aruco_min_marker_depth = 0.1;
  double aruco_max_marker_depth = 10.0;
  double aruco_max_position_residual = 0.6;
  double aruco_max_orientation_residual_deg = 25.0;
  double aruco_position_noise_base = 0.01;
  double aruco_orientation_noise_base = 0.1;
  int aruco_process_stride = 4;
  bool aruco_use_orientation_update = false;
  double aruco_normal_gate_deg = 35.0;
  double aruco_update_max_rot_step_deg = 1.0;
  double aruco_update_max_trans_step_m = 0.08;

  struct BoardObservation 
  {
    int board_id;                     // 地标板子ID
    Eigen::Vector3d center_tvec;      // 地标中心点在相机坐标系下的位置
    Eigen::Matrix3d center_R_cam_board; // 地标中心点到相机的旋转
    int valid_count;                  // 有效的Aruco码数量（固定为4个）
    bool geometry_valid = false;      // 同ID四码几何一致性是否通过
    double center_spread_m = 0.0;     // 四码中心离散度（米）
    double rotation_dispersion_deg = 0.0; // 四码姿态离散度（度）
    //double timestamp;
  };

  struct ArucoObservation 
  {
    int id;
    Eigen::Vector3d tvec;
    Eigen::Matrix3d R_cam_marker;
    //double timestamp;
  };

  std::map<int, bool> board_world_flag_;
  std::map<int, Eigen::Vector3d> board_world_positions_;  // 地标中心点的世界坐标
  std::map<int, Eigen::Matrix3d> board_world_orientations_; // 地标中心点的世界姿态
  std::vector<BoardObservation> current_board_observations_;

  struct BoardConfig 
  {
    double width;     // 宽度（X方向）
    double height;    // 高度（Y方向）
    double marker_size; // Aruco码尺寸 
    double delta_width_qr_center;
    double delta_height_qr_center;
  };
  BoardConfig board_config_;
    
  // 四个Aruco码在板子坐标系下的相对位置
  std::map<int, Eigen::Vector3d> aruco_relative_positions_;

  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;
  int min_retrieve_points = 30;
  int min_update_meas = 600;
  int low_track_force_update_stride = 0;
  int low_track_force_min_points = 8;
  bool deterministic_visual_update_en = true;
  bool deterministic_pixel_snap_en = true;
  bool deterministic_camera_point_snap_en = true;
  bool deterministic_contiguous_image_copy_en = true;
  bool deterministic_visual_voxel_key_sort_en = true;
  bool visual_update_guard_en = true;
  double visual_update_max_trans_m = 0.12;
  double visual_update_max_rot_deg = 2.0;
  double visual_update_max_backward_m = 0.03;
  double visual_update_max_backward_ratio = 0.08;
  double visual_update_backward_abs_floor_m = 0.003;
  double visual_update_max_lateral_m = 0.08;
  double visual_update_max_lateral_ratio = 0.35;
  double visual_update_max_exposure_delta = 0.30;

  double img_point_cov, outlier_threshold, ncc_thre;
  bool image_quality_gate_en = false;
  double image_quality_max_saturated_fraction = 0.20;
  double image_quality_max_tile_saturated_fraction = 0.35;
  double image_quality_max_dark_fraction = 0.98;
  double image_quality_min_intensity_std = 6.0;
  bool visual_patch_quality_gate_en = true;
  double visual_patch_max_saturated_fraction = 0.10;
  double visual_patch_min_intensity_std = 2.0;
  int image_quality_saturated_pixel_value = 250;
  int image_quality_dark_pixel_value = 5;
  int image_quality_tile_rows = 4;
  int image_quality_tile_cols = 4;
  double max_state_update_rot_deg = 0.8;
  double max_state_update_trans_m = 0.08;
  bool visual_map_prune_en = true;
  int visual_map_max_voxels = 12000;
  int visual_map_max_points_per_voxel = 24;
  int visual_map_max_total_points = 180000;
  int visual_map_max_add_per_frame = 600;
  float visual_map_min_shi_tomasi_score = 10.0f;
  double visual_voxel_size = 0.5;
  bool console_timing_print_en = true;
  int console_timing_print_stride = 1;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  double aruco_time_detect_markers = 0.0;
  double aruco_time_draw = 0.0;
  double aruco_time_group_gate = 0.0;
  double aruco_time_pose_estimate = 0.0;
  double aruco_time_pnp = 0.0;
  double aruco_time_update = 0.0;
  double aruco_time_total = 0.0;
  double aruco_ave_time_total = 0.0;
  int aruco_profile_frames = 0;
  int aruco_board_candidates = 0;
  int aruco_board_accepted = 0;
  double last_visual_guard_time = -1.0;
  bool has_last_visual_guard_pos = false;
  V3D last_visual_guard_pos = V3D::Zero();

  string timing_log_dir;
  string timing_log_file_path;
  bool timing_log_enable = true;
  bool timing_log_ready = false;
  int timing_log_flush_stride = 10;
  int timing_log_pending_frames = 0;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  ofstream timing_log_file;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  bool updateStateInverse(cv::Mat img, int level);
  bool updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO(ros::NodeHandle &nh);
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  bool computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  bool insertPointIntoVoxelMap(VisualPoint *pt_new);
  size_t getVisualPointCount() const;
  void pruneVisualMap();
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  void detect_qr(cv::Mat img);
  void draw_qr(std::vector<int>& ids, std::vector<std::vector<cv::Point2f>>& corners, std::vector<std::vector<cv::Point2f>>& rejectedCandidates);
  void updateStateWithBoardObservation();
  Eigen::Matrix3d Exp(const Eigen::Vector3d& w);
  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
  void initializeTimingLogFileIfNeeded();
  void appendTimingLogLines(const vector<string> &lines);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_
