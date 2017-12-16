#include "../test_dropout.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/dropout.hpp"
using namespace nnlib;

void TestDropout()
{
	RandomEngine::sharedEngine().seed(0);
	
	Tensor<NN_REAL_T> ones = Tensor<NN_REAL_T>(4, 25).ones();
	double p = 0.75, sum1 = 0, sum2 = 0;
	size_t c = 100;
	
	Dropout<NN_REAL_T> module(p);
	NNAssertEquals(module.dropProbability(), p, "Dropout::Dropout failed!");
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(ones).sum() / c;
		sum2 += module.backward(ones, ones).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "Dropout::forward failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "Dropout::backward failed!");
	
	module.dropProbability(p = 0.5).training(false);
	sum1 = sum2 = 0;
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(ones).sum() / c;
		sum2 += module.backward(ones, ones).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "Dropout::forward (inference) failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "Dropout::backward (inference) failed!");
	
	module.state().fill(0);
	NNAssertAlmostEquals(module.output().sum(), 0, 1e-12, "Dropout::state failed!");
	
	TestModule("Dropout", module, ones);
}
