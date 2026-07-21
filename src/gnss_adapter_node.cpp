/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#include "gnss_adapter.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gnss_adapter");
  ros::NodeHandle node_handle;
  GnssAdapter adapter;
  if (!adapter.initialize(node_handle)) return 1;
  ros::spin();
  return 0;
}
