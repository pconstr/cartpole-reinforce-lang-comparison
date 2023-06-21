#include "CartPoleEnv.h"

#include <cmath>
#include <cassert>

namespace cartpole {

using std::cos;
using std::sin;

template<class T>
T square(T x) {
  return x*x;
}

Env::Env() : alreadyReset(false), alreadyTerminated(false) {
}

StepResult Env::step(int action) {
  assert(alreadyReset && "Call reset before using step method.");
  assert(!alreadyTerminated && "Don't call step after termination");
  assert((action == 0 || action == 1) && "Action must be 0 (left) or 1 (right)");

  const auto force = (action == 1) ? forceMag : -forceMag;
  const auto costheta = cos(state.theta);
  const auto sintheta = sin(state.theta);

  /*
    For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
  */

  const auto temp = (force + poleMassLength * square(state.thetaDot) * sintheta) / totalMass;

  const auto thetaacc = (gravity * sintheta - costheta * temp)
    /
    (length * (4.0 / 3.0 - masspole * square(costheta) / totalMass))
    ;

  const auto xacc = temp - poleMassLength * thetaacc * costheta / totalMass;

  // compute updated state parameters
  const auto updatedX = state.x + tau * state.xDot;
  const auto updatedXDot = state.xDot + tau * xacc;
  const auto updatedTheta = state.theta + tau * state.thetaDot;
  const auto updatedThetaDot = state.thetaDot + tau * thetaacc;

  // update state
  state.x = updatedX;
  state.xDot = updatedXDot;
  state.theta = updatedTheta;
  state.thetaDot = updatedThetaDot;
        

  alreadyTerminated = (state.x < -xThreshold)
    || (state.x > xThreshold)
    || (state.theta < -thetaThresholdRadians)
    || (state.theta > thetaThresholdRadians)
    ;

  // determine reward

  // terminated means pole just fell, not getting any reward for staying balanced
  // otherwise one more reward point for staying up;
  const float reward = alreadyTerminated? 0.0 : 1.0;
  return StepResult(state, reward, alreadyTerminated);
}

} // namespace cartpole
