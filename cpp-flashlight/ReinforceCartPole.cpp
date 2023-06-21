#include "CartPoleEnv.h"
#include "RollingMean.h"
#include "ChooseAlong.h"
#include "Join.h"

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"

#include <cassert>
#include <iostream>

using namespace cartpole;
using namespace fl;
using namespace util;

using std::array;
using std::back_inserter;
using std::cout;
using std::copy;
using std::discrete_distribution;
using std::endl;
using std::function;
using std::mt19937;
using std::numeric_limits;
using std::random_device;
using std::tuple;
using std::vector;


Tensor sumDiscounted(const Tensor& rewards, float gamma) {
  const int n = rewards.dim(0);
  const auto r = arange(0, n);
  const auto m = power(gamma, r);
  const auto res =  flip(cumsum(flip(m * rewards, 0), 0), 0);
  return res;
}


Tensor packState(const State& state) {
   array<float, 4> stateBuffer = {
     state.x,
     state.xDot,
     state.theta,
     state.thetaDot
   };
   return Tensor::fromBuffer({4, 1}, stateBuffer.data(), MemoryLocation::Host);
}


template<class G>
int act(Sequential& model, const Tensor& stateTensor, G& gen) {
  const auto in = noGrad(stateTensor);
  
  auto logProbs = model(in);
  const auto p = exp(logProbs);
  const auto* rp = p.host<float>();
  discrete_distribution<> discrete(rp, rp + 2);
  int action = discrete(gen);
  return action;
}


Variable recap(Sequential& model, const Tensor& stateTensor, const Tensor& actionTensor) {
  const auto stateIn = noGrad(stateTensor);
  const auto actionIn = noGrad(actionTensor);
  const Variable logProbs = model(stateIn);
  return chooseAlongDim1(logProbs, actionTensor);
}


Sequential buildStochasticPolicyModel() {
  Sequential model;
  model.add(Linear(4, 128));
  model.add(ReLU());
  model.add(Linear(128, 2));
  model.add(LogSoftmax());
  return model;
}


int main(int argc, const char* argv[]) {
  init();

  const float learningRate = 1E-2;
  const float gamma = 0.99;
  const int numberOfEpisodes = 500;
  const int batchSize = 16;

  auto model = buildStochasticPolicyModel();
  /* Build Policy Model */

  auto optim = AdamOptimizer(model.params(),
			     learningRate);

  /* reinforce */

  Env env;
  random_device rd;
  mt19937 gen(rd());

  vector<Tensor> batchedReturns;
  vector<Tensor> batchedStates;
  vector<int> batchedActions;
  int batchedEpisodes = 0;
  int batchCounter = 0;

  RollingMean<float> returnsRollingMean(100);

  int episode = 0;

  auto updateNetwork = [
			&model,
			&optim,
			&batchedReturns,
			&batchedStates,
			&batchedActions
			]() {
    // update network
    const auto returns = join1DTensors(batchedReturns);
    const int n = batchedStates.size();
    assert (n == batchedActions.size());
    const auto states = stack1DTensors(batchedStates);
    const auto actions = Tensor::fromBuffer({n},
					    batchedActions.data(),
					    Location::Host);
    // standardize
    const auto eps = numeric_limits<float>::epsilon();
    const auto m = mean(returns).scalar<float>();
    const auto sd = mean(returns).scalar<float>();
    const auto nreturns = (returns - m) / (sd + eps);

    const auto logProbs = recap(model, states, actions);

    // there is no operator - apparently
    const auto lossTerms = -1 * (logProbs * Variable(nreturns, false));
    auto loss = sum(lossTerms, {0});

    loss.backward();
    optim.step();
    optim.zeroGrad();
  };

  auto report = [&episode,
		 &returnsRollingMean]() {				     
    cout << "* " << episode << " " << returnsRollingMean.mean() << endl;	
  };

  model.eval();

  auto doUpdate = [&model,
		   &batchedReturns,
		   &batchedStates,
		   &batchedActions,
		   &batchedEpisodes,
		   &batchCounter,
		   &updateNetwork,
		   &report] (bool isFinal) {
    model.train();
    updateNetwork();
    model.eval();
    batchedReturns.clear();
    batchedStates.clear();
    batchedActions.clear();
    batchedEpisodes = 0;
    batchCounter = batchCounter + 1;
    if (isFinal || (batchCounter % 10 == 0)) {
      report();
    }
  };

  for(; episode < numberOfEpisodes; ++episode) {
    vector<float> episodeRewards;

    function<void(const State&)> experienceEpisodeRemainder =
      [&env,
       &model,
       &gen,
       &experienceEpisodeRemainder,
       &episodeRewards,
       &batchedStates,
       &batchedActions
       ](const State& observed) {
	// assuming not terminated
	const auto stateTensor = packState(observed);
	const auto action = act(model, stateTensor, gen);
	const auto experienced = env.step(action);

	episodeRewards.push_back(experienced.reward);
	batchedStates.push_back(stateTensor);
	batchedActions.push_back(action);
	
	if (!experienced.terminated) {
	  // tail recursion
	  experienceEpisodeRemainder(experienced.state);
	}
    };

    const auto observed = env.reset(gen);
    experienceEpisodeRemainder(observed);

    const int n = episodeRewards.size();
    const auto vRewards = Tensor::fromBuffer({n},
					      episodeRewards.data(),
					      Location::Host);
    const auto totalReward = sum(vRewards).scalar<float>();
    returnsRollingMean.add(totalReward);
    const auto discounted = sumDiscounted(vRewards, gamma);

    batchedReturns.push_back(discounted);
    batchedEpisodes = batchedEpisodes + 1;

    if (batchedEpisodes == batchSize)
      doUpdate(false);
  }

  if (batchedEpisodes)
    doUpdate(true);
}
