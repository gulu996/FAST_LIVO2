#include "voxel_filter_utils.h"

#include <pcl/point_types.h>

#include <iostream>
#include <limits>

namespace
{
bool check(bool condition, const char *message)
{
  if (condition) return true;
  std::cerr << "FAILED: " << message << std::endl;
  return false;
}

pcl::PointXYZINormal point(float x, float y, float z)
{
  pcl::PointXYZINormal result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.intensity = 1.0f;
  result.curvature = 0.0f;
  return result;
}
}  // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "voxel_filter_self_test", ros::init_options::AnonymousName | ros::init_options::NoSigintHandler);
  ros::Time::init();
  bool passed = true;

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr dirty(new pcl::PointCloud<pcl::PointXYZINormal>());
  dirty->push_back(point(1.00f, 0.0f, 0.0f));
  dirty->push_back(point(1.01f, 0.0f, 0.0f));
  dirty->push_back(point(2.00f, 0.0f, 0.0f));
  dirty->push_back(point(500.0f, 0.0f, 0.0f));
  dirty->push_back(point(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
  dirty->push_back(point(0.0f, std::numeric_limits<float>::infinity(), 0.0f));
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr output(new pcl::PointCloud<pcl::PointXYZINormal>());
  VoxelSafetyContext body_context;
  body_context.coordinate_frame = "lidar/body self-test";
  body_context.enforce_max_range = true;
  body_context.max_range_m = 450.0;
  VoxelDiagnostic diagnostic;
  passed &= check(safeVoxelFilter<pcl::PointXYZINormal>(dirty, output, Eigen::Vector3f::Constant(0.1f),
                                                        "VOXEL_TEST_CLEAN", body_context, &diagnostic),
                  "cleaning filter should succeed");
  passed &= check(diagnostic.invalid_points == 2, "NaN and Inf must be removed");
  passed &= check(diagnostic.extreme_points == 1, "out-of-range body point must be removed");
  passed &= check(output->size() == 2, "valid points must be voxelized after cleaning");

  passed &= check(!safeVoxelFilter<pcl::PointXYZINormal>(dirty, output, Eigen::Vector3f::Zero(),
                                                         "VOXEL_TEST_BAD_LEAF", body_context),
                  "zero leaf must be rejected");
  bool threw = false;
  try
  {
    validateVoxelLeafOrThrow(-0.1, "self_test_leaf");
  }
  catch (const std::runtime_error &)
  {
    threw = true;
  }
  passed &= check(threw, "invalid configured leaf must fail at initialization");

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr wide(new pcl::PointCloud<pcl::PointXYZINormal>());
  wide->push_back(point(1.883f, -43.190f, -6.675f));
  wide->push_back(point(1.884f, -43.189f, -6.674f));
  wide->push_back(point(364.158f, 83.516f, 90.546f));
  VoxelSafetyContext wide_context;
  wide_context.coordinate_frame = "lidar/body HH overflow reproduction";
  wide_context.enforce_max_range = true;
  wide_context.max_range_m = 450.0;
  passed &= check(safeVoxelFilter<pcl::PointXYZINormal>(wide, output, Eigen::Vector3f::Constant(0.1f),
                                                        "VOXEL_TEST_OVERFLOW", wide_context, &diagnostic),
                  "overflow-risk cloud must use the safe fallback");
  passed &= check(diagnostic.overflow_risk, "HH span must be diagnosed as an INT32 overflow risk");
  passed &= check(diagnostic.total_bins > std::numeric_limits<std::int32_t>::max(),
                  "diagnostic total must exceed INT32_MAX without overflowing itself");
  passed &= check(output->size() == 2, "fallback must downsample instead of returning the raw cloud");

  if (!passed) return 1;
  std::cout << "voxel_filter_self_test: PASS" << std::endl;
  return 0;
}
