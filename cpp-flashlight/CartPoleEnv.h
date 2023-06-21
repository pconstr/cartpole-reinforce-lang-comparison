/*
  Classic cart-pole system implemented by Rich Sutton et al.
  adapted from https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/classic_control/cartpole.py
  which was copied from http://incompleteideas.net/sutton/book/code/pole.c
*/

#pragma once

#include <random>

namespace cartpole {

/* transparent POD representing the (fully observable) state */
struct State {
  float x; // Cart position
  float xDot ; // Cart velocity
  float theta; // Pole angle
  float thetaDot; // Pole angular velocity;
};

struct StepResult {
  State state;
  float reward;
  bool terminated;

StepResult(const State& state, float reward, bool terminated):
  state(state), reward(reward), terminated(terminated) {}
};

class Env {
  /*
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
  */

public:
  Env();

  StepResult step(int action);

  template<class G>
    State reset(G& gen) {
    /* same for all parameters as in original code */
    std::uniform_real_distribution<> dis(-0.05, 0.05);
    state.x = dis(gen);
    state.xDot = dis(gen);
    state.theta = dis(gen);
    state.thetaDot = dis(gen);

    alreadyReset = true;
    alreadyTerminated = false;
    return state;    
  }

private:
  State state;

  static constexpr double pi = 3.14159265358979323846;

  static constexpr double gravity = 9.8f;
  static constexpr double masscart = 1.0;
  static constexpr double masspole = 0.1;
  static constexpr double totalMass = masspole + masscart;
  static constexpr double length = 0.5; // actually half the pole's length
  static constexpr double poleMassLength = masspole * length;
  static constexpr double forceMag = 10.0;
  static constexpr double tau = 0.02;
  // Angle at which to fail the episode
  static constexpr double thetaThresholdRadians = 12 * 2 *pi / 360;
  static constexpr double xThreshold = 2.4;

  bool alreadyTerminated;
  bool alreadyReset;

  // reward given immediately after arriving at current state
  float immediateReward() const;
};

} // namespace cartpole
