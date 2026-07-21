#ifndef VOXEL_FILTER_UTILS_H
#define VOXEL_FILTER_UTILS_H

#include <Eigen/Core>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

struct VoxelDiagnostic
{
  std::size_t input_points = 0;
  std::size_t finite_points = 0;
  std::size_t invalid_points = 0;
  std::size_t extreme_points = 0;
  Eigen::Vector3d min_point = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_point = Eigen::Vector3d::Zero();
  Eigen::Vector3d span = Eigen::Vector3d::Zero();
  Eigen::Vector3d leaf_size = Eigen::Vector3d::Zero();
  std::uint64_t bins_x = 0;
  std::uint64_t bins_y = 0;
  std::uint64_t bins_z = 0;
  long double total_bins = 0.0L;
  double max_distance_m = 0.0;
  bool overflow_risk = false;
};

struct VoxelSafetyContext
{
  std::string coordinate_frame = "unknown";
  bool enforce_max_range = false;
  double max_range_m = std::numeric_limits<double>::infinity();
  Eigen::Vector3d range_origin = Eigen::Vector3d::Zero();
};

inline void validateVoxelLeafOrThrow(double leaf_size, const std::string &parameter_name)
{
  if (!std::isfinite(leaf_size) || leaf_size <= 0.0)
  {
    throw std::runtime_error("Invalid voxel leaf size for " + parameter_name + ": " + std::to_string(leaf_size));
  }
}

namespace voxel_filter_detail
{
struct VoxelKey
{
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const VoxelKey &other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey &key) const
  {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

inline std::uint64_t axisBinCount(double span, double leaf)
{
  const long double bins = std::floor(static_cast<long double>(span) / static_cast<long double>(leaf)) + 1.0L;
  if (bins >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
    return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(bins);
}

inline bool voxelIndex(double coordinate, double leaf, std::int64_t &index)
{
  const long double value = std::floor(static_cast<long double>(coordinate) / static_cast<long double>(leaf));
  if (value < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      value > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
    return false;
  index = static_cast<std::int64_t>(value);
  return true;
}
}  // namespace voxel_filter_detail

template <typename PointT>
bool safeVoxelFilter(const typename pcl::PointCloud<PointT>::ConstPtr &input,
                     const typename pcl::PointCloud<PointT>::Ptr &output,
                     const Eigen::Vector3f &leaf_size,
                     const std::string &tag,
                     const VoxelSafetyContext &context,
                     VoxelDiagnostic *diagnostic_out = nullptr)
{
  if (!output)
  {
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=null_output");
    return false;
  }
  output->clear();

  if (!input)
  {
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=null_input");
    return false;
  }
  if (!leaf_size.allFinite() || (leaf_size.array() <= 0.0f).any())
  {
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=invalid_leaf leaf=["
                                      << leaf_size.transpose() << "]");
    return false;
  }
  if (context.enforce_max_range && (!std::isfinite(context.max_range_m) || context.max_range_m <= 0.0))
  {
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=invalid_max_range value="
                                      << context.max_range_m);
    return false;
  }

  VoxelDiagnostic diagnostic;
  diagnostic.input_points = input->size();
  diagnostic.leaf_size = leaf_size.cast<double>();
  Eigen::Vector3d min_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d max_point = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  const double max_range_sq = context.max_range_m * context.max_range_m;

  typename pcl::PointCloud<PointT>::Ptr cleaned(new pcl::PointCloud<PointT>());
  cleaned->header = input->header;
  cleaned->reserve(input->size());
  for (const PointT &point : input->points)
  {
    const Eigen::Vector3d xyz(point.x, point.y, point.z);
    if (!xyz.allFinite())
    {
      ++diagnostic.invalid_points;
      continue;
    }
    ++diagnostic.finite_points;
    const double distance_sq = (xyz - context.range_origin).squaredNorm();
    diagnostic.max_distance_m = std::max(diagnostic.max_distance_m, std::sqrt(distance_sq));
    if (context.enforce_max_range && distance_sq > max_range_sq)
    {
      ++diagnostic.extreme_points;
      continue;
    }
    min_point = min_point.cwiseMin(xyz);
    max_point = max_point.cwiseMax(xyz);
    cleaned->push_back(point);
  }
  cleaned->width = static_cast<std::uint32_t>(cleaned->size());
  cleaned->height = 1;
  cleaned->is_dense = true;

  if (cleaned->empty())
  {
    if (diagnostic_out) *diagnostic_out = diagnostic;
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=no_valid_points input="
                                      << diagnostic.input_points << " invalid=" << diagnostic.invalid_points
                                      << " extreme=" << diagnostic.extreme_points << " frame=" << context.coordinate_frame);
    return false;
  }

  diagnostic.min_point = min_point;
  diagnostic.max_point = max_point;
  diagnostic.span = max_point - min_point;
  diagnostic.bins_x = voxel_filter_detail::axisBinCount(diagnostic.span.x(), diagnostic.leaf_size.x());
  diagnostic.bins_y = voxel_filter_detail::axisBinCount(diagnostic.span.y(), diagnostic.leaf_size.y());
  diagnostic.bins_z = voxel_filter_detail::axisBinCount(diagnostic.span.z(), diagnostic.leaf_size.z());
  diagnostic.total_bins = static_cast<long double>(diagnostic.bins_x) *
                          static_cast<long double>(diagnostic.bins_y) *
                          static_cast<long double>(diagnostic.bins_z);

  const long double int32_max = static_cast<long double>(std::numeric_limits<std::int32_t>::max());
  diagnostic.overflow_risk = diagnostic.total_bins > int32_max;
  for (int axis = 0; axis < 3; ++axis)
  {
    const long double min_index = std::floor(static_cast<long double>(min_point[axis]) / diagnostic.leaf_size[axis]);
    const long double max_index = std::floor(static_cast<long double>(max_point[axis]) / diagnostic.leaf_size[axis]);
    if (min_index < static_cast<long double>(std::numeric_limits<std::int32_t>::min()) ||
        max_index > static_cast<long double>(std::numeric_limits<std::int32_t>::max()))
      diagnostic.overflow_risk = true;
  }

  bool used_sparse_fallback = false;
  if (!diagnostic.overflow_risk)
  {
    pcl::VoxelGrid<PointT> filter;
    filter.setInputCloud(cleaned);
    filter.setLeafSize(leaf_size.x(), leaf_size.y(), leaf_size.z());
    filter.filter(*output);
  }
  else
  {
    used_sparse_fallback = true;
    ROS_WARN_STREAM_THROTTLE(5.0, "[VOXEL_OVERFLOW_RISK] tag=" << tag << " points=" << cleaned->size()
                                     << " min=[" << diagnostic.min_point.transpose() << "] max=["
                                     << diagnostic.max_point.transpose() << "] span=[" << diagnostic.span.transpose()
                                     << "] leaf=[" << diagnostic.leaf_size.transpose() << "] bins=["
                                     << diagnostic.bins_x << " " << diagnostic.bins_y << " " << diagnostic.bins_z
                                     << "] total_bins=" << diagnostic.total_bins << " frame=" << context.coordinate_frame
                                     << " action=sparse_voxel_fallback");

    std::unordered_map<voxel_filter_detail::VoxelKey, pcl::CentroidPoint<PointT>, voxel_filter_detail::VoxelKeyHash> voxels;
    voxels.reserve(cleaned->size());
    for (const PointT &point : cleaned->points)
    {
      voxel_filter_detail::VoxelKey key;
      if (!voxel_filter_detail::voxelIndex(point.x, leaf_size.x(), key.x) ||
          !voxel_filter_detail::voxelIndex(point.y, leaf_size.y(), key.y) ||
          !voxel_filter_detail::voxelIndex(point.z, leaf_size.z(), key.z))
      {
        output->clear();
        if (diagnostic_out) *diagnostic_out = diagnostic;
        ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=voxel_index_exceeds_int64");
        return false;
      }
      voxels[key].add(point);
    }
    output->reserve(voxels.size());
    for (auto &entry : voxels)
    {
      PointT centroid;
      entry.second.get(centroid);
      output->push_back(centroid);
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = true;
  }

  bool output_valid = !output->empty() && output->size() <= cleaned->size();
  for (const PointT &point : output->points)
    output_valid = output_valid && std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
  if (!output_valid)
  {
    output->clear();
    if (diagnostic_out) *diagnostic_out = diagnostic;
    ROS_ERROR_STREAM_THROTTLE(5.0, "[VOXEL_REJECT] tag=" << tag << " reason=invalid_filter_output");
    return false;
  }

  if (diagnostic_out) *diagnostic_out = diagnostic;
  const double reduction_ratio = 1.0 - static_cast<double>(output->size()) /
                                           static_cast<double>(diagnostic.input_points);
  ROS_INFO_STREAM_THROTTLE(5.0, "[VOXEL_STATS] tag=" << tag << " input=" << diagnostic.input_points
                                   << " finite=" << diagnostic.finite_points
                                   << " removed_invalid=" << diagnostic.invalid_points
                                   << " removed_extreme=" << diagnostic.extreme_points << " output=" << output->size()
                                   << " leaf=[" << diagnostic.leaf_size.transpose() << "] span=["
                                   << diagnostic.span.transpose() << "] max_distance_m=" << diagnostic.max_distance_m
                                   << " reduction_ratio=" << reduction_ratio << " fallback=" << used_sparse_fallback
                                   << " frame=" << context.coordinate_frame);
  return true;
}

#endif
