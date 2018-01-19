#include "../test_lstm.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/lstm.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(LSTM)
{
    NNRunAbstractTest(Module, LSTM, new LSTM<T>(3, 2));

    NNTestMethod(LSTM)
    {
        NNTestParams(size_t, size_t)
        {
            LSTM<T> module(3, 2);
            NNTestEquals(module.inputs(), 3);
            NNTestEquals(module.outputs(), 2);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(LSTM)
        {
            LSTM<T> orig(3, 2);
            LSTM<T> copy(1, 1);
            copy = orig;
            NNTestEquals(copy.inputs(), 3);
            NNTestEquals(copy.outputs(), 2);
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(gradClip)
    {
        NNTestParams(T)
        {
            LSTM<T> module(1, 1);
            module.params().copy({
                -0.2, 0.5, 0.1, 0,
                0.75, -0.6, 0.25, 0,
                1.0, -0.7, 0,
                0.3, 0.3, -0.75, 0
            });
            module.gradClip(0.2);
            NNTestAlmostEquals(module.gradClip(), 0.2, 1e-12);

            auto input = Tensor<T>({ 8, 6, 0 }).resize(3, 1, 1);
            auto blame = Tensor<T>({ 1, 0, -1 }).resize(3, 1, 1);
            auto inGrad = Tensor<T>({ -0.01712796895, 0.00743178473, -0.2 });

            Tensor<T> actual(3);
            Tensor<T> states(2, module.state().size());

            module.forward(input.select(0, 0))(0, 0);
            states.select(0, 0).copy(module.state());
            module.forward(input.select(0, 1))(0, 0);
            states.select(0, 1).copy(module.state());
            module.forward(input.select(0, 2))(0, 0);

            actual(2) = module.backward(input.select(0, 2), blame.select(0, 2))(0, 0);
            module.state().copy(states.select(0, 1));
            actual(1) = module.backward(input.select(0, 1), blame.select(0, 1))(0, 0);
            module.state().copy(states.select(0, 0));
            actual(0) = module.backward(input.select(0, 0), blame.select(0, 0))(0, 0);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, actual, inGrad);
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            LSTM<T> module(1, 1);
            module.params().copy({
                -0.2, 0.5, 0.1, 0,
                0.75, -0.6, 0.25, 0,
                1.0, -0.7, 0,
                0.3, 0.3, -0.75, 0
            });

            auto input = Tensor<T>({ 8, 6, 0 }).resize(3, 1, 1);
            auto target = Tensor<T>({ 0.15089258930, 0.32260369939, 0.03848645247 });

            Tensor<T> outputs(3);
            outputs(0) = module.forward(input.select(0, 0))(0, 0);
            outputs(1) = module.forward(input.select(0, 1))(0, 0);
            outputs(2) = module.forward(input.select(0, 2))(0, 0);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, outputs, target);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            LSTM<T> module(1, 1);
            module.params().copy({
                -0.2, 0.5, 0.1, 0,
                0.75, -0.6, 0.25, 0,
                1.0, -0.7, 0,
                0.3, 0.3, -0.75, 0
            });

            auto input = Tensor<T>({ 8, 6, 0 }).resize(3, 1, 1);
            auto blame = Tensor<T>({ 1, 0, -1 }).resize(3, 1, 1);
            auto inGrad = Tensor<T>({ -0.01712796895, 0.00743178473, -0.30729831287 });
            auto pGrad = Tensor<T>({
                0.73850659626, 0.00585685005, 0.00801518653, 0.11462669318,
                -0.00117516696, -0.01646944995, -0.02114722491, -0.05115589662,
                -0.00000416626, -0.08323296026, -0.25800409062,
                0.18717939172, -0.00419251188, 0.00611825134, 0.00767284875
            });

            Tensor<T> actual(3);
            Tensor<T> states(2, module.state().size());

            module.forward(input.select(0, 0))(0, 0);
            states.select(0, 0).copy(module.state());
            module.forward(input.select(0, 1))(0, 0);
            states.select(0, 1).copy(module.state());
            module.forward(input.select(0, 2))(0, 0);

            actual(2) = module.backward(input.select(0, 2), blame.select(0, 2))(0, 0);
            module.state().copy(states.select(0, 1));
            actual(1) = module.backward(input.select(0, 1), blame.select(0, 1))(0, 0);
            module.state().copy(states.select(0, 0));
            actual(0) = module.backward(input.select(0, 0), blame.select(0, 0))(0, 0);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, actual, inGrad);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.grad(), pGrad);
        }
    }
}
