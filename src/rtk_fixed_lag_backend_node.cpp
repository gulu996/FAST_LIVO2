#include "rtk_fixed_lag_backend.h"

#include <exception>

int main(int argc, char **argv) {
  ros::init(argc, argv, "rtk_fixed_lag_backend");
  ros::NodeHandle node;
  try {
    fast_livo_backend::RtkFixedLagBackend backend(node);
    if (!backend.enabled()) return 0;
    ros::spin();
  } catch (const std::exception &error) {
    ROS_FATAL_STREAM("[RTK_BACKEND] startup failed: " << error.what());
    return 1;
  }
  return 0;
}
