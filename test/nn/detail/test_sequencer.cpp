#include "../test_module.hpp"
#include "../test_sequencer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/nn/sequencer.hpp"
#include "nnlib/nn/lstm.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Sequencer)
{
    NNRunAbstractTest(Module, Sequencer, new Sequencer<T>(new LSTM<T>(3, 2)));

    NNTestMethod(Sequencer)
    {
        NNTestParams(Module *)
        {
            auto lstm = new LSTM<T>(3, 2);
            Sequencer<T> module(lstm);
            NNTestEquals(&module.module(), lstm);
            NNTest(!module.isReversed());
        }

        NNTestParams(Module *, bool)
        {
            auto lstm = new LSTM<T>(3, 2);
            Sequencer<T> module(lstm, true);
            NNTestEquals(&module.module(), lstm);
            NNTest(module.isReversed());
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(LSTM)
        {
            Sequencer<T> orig(new LSTM<T>(3, 2), true);
            Sequencer<T> copy(new LSTM<T>(1, 1));
            copy = orig;
            NNTest(copy.isReversed());
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(module)
    {
        NNTestParams(Module *)
        {
            auto lstm1 = new LSTM<T>(3, 2);
            auto lstm2 = new LSTM<T>(2, 1);
            Sequencer<T> module(lstm1);
            module.module(lstm2);
            NNTestEquals(&module.module(), lstm2);
        }
    }

    NNTestMethod(reverse)
    {
        NNTestParams(bool)
        {
            Sequencer<T> orig(new LSTM<T>(3, 2));
            orig.reverse(true);
            NNTest(orig.isReversed());
            orig.reverse(false);
            NNTest(!orig.isReversed());
        }
    }

    NNTestMethod(training)
    {
        NNTestParams(bool)
        {
            auto component = new BatchNorm<T>(10);
            Sequencer<T> module(component);
            module.training(true);
            NNTest(component->isTraining());
            module.training(false);
            NNTest(!component->isTraining());
        }
    }

    NNTestMethod(forget)
    {
        NNTestParams()
        {
            auto component = new LSTM<T>(3, 2);
            Sequencer<T> module(component);
            component->state().fill(1);
            module.forget();
            forEach([&](T state)
            {
                NNTestAlmostEquals(state, 0, 1e-12);
            }, component->state());
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            Sequencer<T> module(new LSTM<T>(1, 1));
            module.params().copy({
                -0.2, 0.5, 0.1, 0,
                0.75, -0.6, 0.25, 0,
                1.0, -0.7, 0,
                0.3, 0.3, -0.75, 0
            });

            auto input = Tensor<T>({ 8, 6, 0 }).resize(3, 1, 1);
            auto target = Tensor<T>({ 0.15089258930, 0.32260369939, 0.03848645247 }).resize(3, 1, 1);

            module.forward(input);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.output(), target);

            target.copy({ 0.3523252224669841, 0.1900534836059371, 0 });
            module.forget();
            module.reverse(true);
            module.forward(input);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.output(), target);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Sequencer<T> module(new LSTM<T>(1, 1));
            module.params().copy({
                -0.2, 0.5, 0.1, 0,
                0.75, -0.6, 0.25, 0,
                1.0, -0.7, 0,
                0.3, 0.3, -0.75, 0
            });

            auto input = Tensor<T>({ 8, 6, 0 }).resize(3, 1, 1);
            auto blame = Tensor<T>({ 1, 0, -1 }).resize(3, 1, 1);
            auto inGrad = Tensor<T>({ -0.01712796895, 0.00743178473, -0.30729831287 }).resize(3, 1, 1);
            auto pGrad = Tensor<T>({
                0.73850659626, 0.00585685005, 0.00801518653, 0.11462669318,
                -0.00117516696, -0.01646944995, -0.02114722491, -0.05115589662,
                -0.00000416626, -0.08323296026, -0.25800409062,
                0.18717939172, -0.00419251188, 0.00611825134, 0.00767284875
            });

            module.forward(input);
            module.backward(input, blame);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.inGrad(), inGrad);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.grad(), pGrad);

            inGrad.copy({ -0.01058507197696686, -0.02743768942521932, 0.1646924317207443 });
            module.forget();
            module.reverse(true);
            module.forward(input);
            module.backward(input, blame);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-9);
            }, module.inGrad(), inGrad);
        }
    }
}
