#include "uwb_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  const V3D anchor0(-0.1, -0.4, 0.7);
  const V3D tag0(-0.00295444, -0.000866324, -0.00123799);

  const double predicted_3d = uwbPredictedRange3d(tag0, anchor0);
  const double predicted_xy = uwbPredictedRangeXy(tag0, anchor0);

  assert(std::fabs(predicted_3d - 0.813) < 1e-3);
  assert(std::fabs(predicted_xy - 0.411) < 1e-3);
  assert(predicted_3d - predicted_xy > 0.4);

  std::cout << "predicted_3d=" << predicted_3d
            << " predicted_xy=" << predicted_xy << std::endl;
  return 0;
}
