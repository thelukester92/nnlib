#include "../test_module.hpp"
#include "../test_map.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Map, Map<T>)
{
    NNRunAbstractTest(Module, Map, nnImpl.copy());

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Serialized s;
            nnImpl.save(s);

            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));

            auto copy = Serialized(nnImpl).get<Map<T> *>();

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
}
