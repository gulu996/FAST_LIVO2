#include "LIVMapper.h"
#include <Eigen/Core>
#include <cstdlib>
#include <opencv2/core.hpp>

int main(int argc, char **argv)
{
  /*if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn)) 
  {
    ros::console::notifyLoggerLevelsChanged();
  }*/
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  bool force_single_thread = false;
  nh.param<bool>("deterministic_debug/force_single_thread", force_single_thread, false);
  if (force_single_thread)
  {
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);
    Eigen::setNbThreads(1);
    cv::setNumThreads(1);
  }

  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}
