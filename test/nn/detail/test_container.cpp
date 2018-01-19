#include "../test_module.hpp"
#include "../test_container.hpp"
#include "nnlib/nn/batchnorm.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Container, Container<T>)
{
    NNRunAbstractTest(Module, Container, nnImpl.copy());

    NNTestMethod(training)
    {
        auto component = new BatchNorm<T>(10);
        auto copy = (Container<T> *) nnImpl.copy();
        copy->clear();
        copy->add(component);
        copy->training(true);
        NNTest(component->isTraining());
        copy->training(false);
        NNTest(!component->isTraining());
        delete copy;
    }

    NNTestMethod(forget)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
            {
                Module<T> &component = *nnImpl.component(i);
                component.state().fill(i);
                auto oldState = component.state().copy();
                nnImpl.forget();
                forEach([&](T oldState, T newState)
                {
                    NNTestAlmostEquals(oldState, i, 1e-12);
                    NNTestAlmostEquals(newState, 0, 1e-12);
                }, oldState, component.state());
            }
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Serialized s;
            nnImpl.save(s);

            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));

            auto copy = Serialized(nnImpl).get<Container<T> *>();
            NNTestEquals(nnImpl.components(), copy->components());

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            nnImpl.grad().fill(0);
            nnImpl.forward(input);
            nnImpl.backward(input, output);

            RandomEngine::sharedEngine().seed(0);
            copy->forget();
            copy->grad().fill(0);
            copy->forward(input);
            copy->backward(input, output);

            forEach([&](T origOutput, T copyOutput)
            {
                NNTestAlmostEquals(origOutput, copyOutput, 1e-12);
            }, nnImpl.output(), copy->output());

            forEach([&](T origInGrad, T copyInGrad)
            {
                NNTestAlmostEquals(origInGrad, copyInGrad, 1e-12);
            }, nnImpl.inGrad(), copy->inGrad());

            forEach([&](T origParamGrad, T copyParamGrad)
            {
                NNTestAlmostEquals(origParamGrad, copyParamGrad, 1e-12);
            }, nnImpl.grad(), copy->grad());

            delete copy;
        }
    }

    NNTestMethod(add)
    {
        NNTestParams(Module *)
        {
            size_t count = nnImpl.components();
            auto last = nnImpl.remove(count - 1);
            NNTestEquals(count - 1, nnImpl.components());
            nnImpl.add(last);
            NNTestEquals(count, nnImpl.components());
        }
    }

    NNTestMethod(remove)
    {
        NNTestParams(Module *)
        {
            size_t count = nnImpl.components();
            auto last = nnImpl.remove(count - 1);
            NNTestEquals(count - 1, nnImpl.components());
            nnImpl.add(last);
            NNTestEquals(count, nnImpl.components());
        }
    }

    NNTestMethod(clear)
    {
        NNTestParams(Module *)
        {
            auto copy = (Container<T> *) nnImpl.copy();
            copy->clear();
            NNTestEquals(copy->components(), 0);
            delete copy;
        }
    }

    NNTestMethod(paramsList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->params().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.params());
        }
    }

    NNTestMethod(gradList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->grad().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.grad());
        }
    }

    NNTestMethod(stateList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->state().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.state());
        }
    }
}
