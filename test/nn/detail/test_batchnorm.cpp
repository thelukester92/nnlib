#include "../test_batchnorm.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(BatchNorm)
{
    NNRunAbstractTest(Module, BatchNorm, new BatchNorm<T>(10));

    NNTestMethod(BatchNorm)
    {
        NNTestParams(size_t)
        {
            BatchNorm<T> module(5);
            NNTest(module.isTraining());
            NNTestEquals(module.inputShape()[1], 5);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(BatchNorm)
        {
            BatchNorm<T> orig(10);
            BatchNorm<T> copy(25);
            copy = orig;

            NNTestEquals(orig.inputShape(), copy.inputShape());
            NNTestEquals(orig.outputShape(), copy.outputShape());

            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            BatchNorm<T> module(10);
            module.weights().fill(10);
            module.bias().fill(5);
            module.reset();

            forEach([&](T weight)
            {
                NNTestGreaterThanOrEquals(weight, -1);
                NNTestLessThanOrEquals(weight, 1);
            }, module.weights());

            forEach([&](T bias)
            {
                NNTestAlmostEquals(bias, 0, 1e-12);
            }, module.bias());
        }
    }

    NNTestMethod(momentum)
    {
        NNTestParams(T)
        {
            BatchNorm<T> module(10);
            module.momentum(0.5);
            NNTestAlmostEquals(module.momentum(), 0.5, 1e-12);
            module.momentum(0.1);
            NNTestAlmostEquals(module.momentum(), 0.1, 1e-12);
        }
    }

    NNTestMethod(training)
    {
        NNTestParams(bool)
        {
            BatchNorm<T> module(10);
            NNTest(module.isTraining());
            module.training(false);
            NNTest(!module.isTraining());
            module.training(true);
            NNTest(module.isTraining());
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            BatchNorm<T> module(10);
            module.weights().fill(1);
            module.bias().fill(2);
            module.momentum(0.125);
            module.training(false);

            Serialized s;
            module.save(s);

            forEach([&](T weight)
            {
                NNTestAlmostEquals(weight, 1, 1e-12);
            }, s.get<Tensor<T>>("weights"));

            forEach([&](T bias)
            {
                NNTestAlmostEquals(bias, 2, 1e-12);
            }, s.get<Tensor<T>>("biases"));

            NNTestAlmostEquals(s.get<T>("momentum"), 0.125, 1e-12);
            NNTest(!s.get<bool>("training"));
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            auto input = Tensor<T>({ 3, 6, 9, -1, 5, 4, 12, 5, 11 }).resize(3, 3);

            BatchNorm<T> module(3);
            module.weights().ones();
            module.bias().zeros();
            module.momentum(1.0);

            module.forward(input);
            for(size_t i = 0; i < 3; ++i)
            {
                NNTestAlmostEquals(math::mean(module.output().select(1, i)), 0, 1e-9);
                NNTestAlmostEquals(math::variance(module.output().select(1, i)), 1, 1e-9);
            }

            module.training(false);

            module.forward(input);
            for(size_t i = 0; i < 3; ++i)
            {
                NNTestAlmostEquals(math::mean(module.output().select(1, i)), 0, 1e-9);
                NNTestAlmostEquals(math::variance(module.output().select(1, i).scale(sqrt(3) / sqrt(2))), 1, 1e-9);
            }
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {

        }
    }

    NNTestMethod(paramsList)
    {
        NNTestParams()
        {
            // weights, biases
        }
    }

    NNTestMethod(gradList)
    {
        NNTestParams()
        {
            // weightsGrad, biasesGrad
        }
    }

    NNTestMethod(stateList)
    {
        NNTestParams()
        {
            // means, invStds, runningMeans, runningVars
        }
    }
}

/*
void TestBatchNorm()
{
    Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({
         3,  6,  9,
        -1,  5,  4,
        12,  5, 11
    }).resize(3, 3);

    Tensor<NN_REAL_T> grad = Tensor<NN_REAL_T>({
         2,  3,  4,
        -2,  0,  4,
        10,  2,  4
    }).resize(3, 3);

    Tensor<NN_REAL_T> inGrad = Tensor<NN_REAL_T>({
         0.03596,  0.00000,  0,
        -0.02489, -2.12132,  0,
        -0.01106,  2.12132,  0
    }).resize(3, 3);

    Tensor<NN_REAL_T> paramGrad = Tensor<NN_REAL_T>({
        14.9606, 2.82843, 0,
        10, 5, 12
    });

    BatchNorm<NN_REAL_T> bn(3);
    bn.weights().ones();
    bn.bias().zeros();
    bn.momentum(1.0);

    bn.forward(inp);
    for(size_t i = 0; i < 3; ++i)
    {
        NNAssertLessThan(fabs(math::mean(bn.output().select(1, i))), 1e-9, "BatchNorm::forward failed! Non-zero mean!");
        NNAssertAlmostEquals(math::variance(bn.output().select(1, i)), 1, 1e-9, "BatchNorm::forward failed! Non-unit variance!");
    }

    bn.backward(inp, grad);
    NNAssertLessThan(math::sum(math::square(bn.grad() - paramGrad)), 1e-9, "BatchNorm::backward failed! Wrong parameter gradient!");
    NNAssertLessThan(math::sum(math::square(bn.inGrad() - inGrad)), 1e-9, "BatchNorm::backward failed! Wrong input gradient!");

    bn.training(false);

    paramGrad.copy({
        27.17588, 5.13783, 0,
        20, 10, 24
    });

    inGrad.copy({
         0.30038, 5.19615, 1.10940,
        -0.30038, 0.00000, 1.10940,
         1.50188, 3.46410, 1.10940
    });

    bn.forward(inp);
    for(size_t i = 0; i < 3; ++i)
    {
        NNAssertLessThan(fabs(math::mean(bn.output().select(1, i))), 1e-9, "BatchNorm::forward (inference) failed! Non-zero mean!");
        NNAssertAlmostEquals(
            math::variance(bn.output().select(1, i).scale(sqrt(3) / sqrt(2))), 1, 1e-9,
            "BatchNorm::forward (inference) failed! Non-unit variance!"
        );
    }

    bn.backward(inp, grad);
    NNAssertLessThan(math::sum(math::square(bn.grad() - paramGrad)), 1e-9, "BatchNorm::backward (inference) failed! Wrong parameter gradient!");
    NNAssertLessThan(math::sum(math::square(bn.inGrad() - inGrad)), 1e-9, "BatchNorm::backward (inference) failed! Wrong input gradient!");

    Tensor<NN_REAL_T> state = bn.state();
    state.fill(0);
    NNAssertAlmostEquals(math::sum(bn.output()), 0, 1e-12, "BatchNorm::state failed!");

    TestModule("BatchNorm", bn, inp);
}
*/
