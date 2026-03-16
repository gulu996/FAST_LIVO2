#include "LIVMapper.h"

int main(int argc, char **argv)
{
  /*if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn)) 
  {
    ros::console::notifyLoggerLevelsChanged();
  }*/
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}