#include "../test_criticsequencer.hpp"
#include "nnlib/critics/criticsequencer.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;

void TestCriticSequencer()
{
	Storage<size_t> shape = { 5, 1, 1 };
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<NN_REAL_T> tgt = Tensor<NN_REAL_T>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<NN_REAL_T> sqd = Tensor<NN_REAL_T>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<NN_REAL_T> dif = Tensor<NN_REAL_T>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<NN_REAL_T> *innerCritic = new MSE<NN_REAL_T>(false);
	CriticSequencer<NN_REAL_T> critic(innerCritic);

	double mse = critic.forward(inp, tgt);
	NNAssertAlmostEquals(mse, math::sum(sqd), 1e-12, "CriticSequencer::forward with no average failed!");

	critic.backward(inp, tgt);
	NNAssert(math::sum(math::square(critic.inGrad().reshape(5, 1) - dif.reshape(5, 1))) < 1e-12, "CriticSequencer::backward failed!");
}
