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
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("OMP_DYNAMIC", "FALSE", 1);
  Eigen::setNbThreads(1);
  cv::setNumThreads(1);

  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}
