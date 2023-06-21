#include "CartPoleEnv.h"
#include "RollingMean.h"

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


Tensor join1DTensors(const vector<Tensor>& tensors) {
  int totalSize = 0;
  for (auto t: tensors)
    totalSize += t.dim(0);
  Tensor out({totalSize});//, arrays[0].type());
  int i = 0;
  for (auto t: tensors) {
    const auto l = t.dim(0);
    out(range(i, i+l)) = t;
    i += l;
  }
  return out;
}


template<class G>
tuple<int, Variable> act(Sequential& model, const State& state, G& gen) {
  array<float, 4> stateBuffer = {
    state.x,
    state.xDot,
    state.theta,
    state.thetaDot
  };
  Tensor inputTensor = Tensor::fromBuffer({4, 1}, stateBuffer.data(), MemoryLocation::Host);
  const auto in = noGrad(inputTensor);
  auto logProbs = model(in);
  const auto p = exp(logProbs);
  const auto* rp = p.host<float>();
  discrete_distribution<> discrete(rp, rp + 2);
  int action = discrete(gen);
  const Variable logProbAction = logProbs(action);
  return {action, logProbAction};
}


int main(int argc, const char* argv[]) {
  init();

  const float learningRate = 1E-2;
  const float gamma = 0.99;
  const int numberOfEpisodes = 1000;
  const int batchSize = 64;

  /* Build Policy Model */

  Sequential model;
  model.add(Linear(4, 128));
  model.add(ReLU());
  model.add(Linear(128, 2));
  model.add(LogSoftmax());

  auto optim = AdamOptimizer(model.params(),
			     learningRate);

  /* reinforce */

  Env env;
  random_device rd;
  mt19937 gen(rd());

  vector<Tensor> batchedReturns;
  vector<Variable> batchedLogProbs;
  int batchedEpisodes = 0;
  int batchCounter = 0;

  RollingMean<float> returnsRollingMean(100);

  int episode = 0;

  auto updateNetwork = [
			&optim,
			&batchedReturns,
			&batchedLogProbs
			]() {
    // update network
    const auto returns = join1DTensors(batchedReturns);
    // standardize
    const auto eps = numeric_limits<float>::epsilon();
    const auto m = mean(returns).scalar<float>();
    const auto sd = mean(returns).scalar<float>();
    const auto nreturns = (returns - m) / (sd + eps);

    const auto logProbs = concatenate(batchedLogProbs, 0);
    // there is no operator - apparently
    const auto lossTerms = -1 * (logProbs * Variable(nreturns, false));
    auto loss = sum(lossTerms, {0});

    loss.backward();
    optim.step();
    optim.zeroGrad();
  };

  auto report = [&episode,
		 &returnsRollingMean,
		 &batchedReturns]() {				     
    cout << "* " << episode << " " << returnsRollingMean.mean() << endl;	
  };

  for(; episode < numberOfEpisodes; ++episode) {
    vector<float> episodeRewards;
    vector<Variable> episodeLogProbs;

    function<void(const State&)> experienceEpisodeRemainder =
      [&env,
       &model,
       &gen,
       &experienceEpisodeRemainder,
       &episodeRewards,
       &episodeLogProbs
       ](const State& observed) {
	// assuming not terminated
	auto [action, logProbAction] = act(model, observed, gen);
	const auto experienced = env.step(action);

	episodeRewards.push_back(experienced.reward);
	episodeLogProbs.push_back(logProbAction);
	
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
    copy(episodeLogProbs.begin(), episodeLogProbs.end(), back_inserter(batchedLogProbs));
    batchedEpisodes = batchedEpisodes + 1;

    if (batchedEpisodes == batchSize) {
      updateNetwork();
      batchedReturns.clear();
      batchedLogProbs.clear();
      batchedEpisodes = 0;
      batchCounter = batchCounter + 1;
      
      if (batchCounter % 10 == 0) {
	report();
      }      
    }
  }

  if (batchedEpisodes) {
    updateNetwork();
    report();
  }
}
