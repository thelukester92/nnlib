#include "../test_tensor_util.hpp"
#include "nnlib/core/detail/tensor_util.hpp"
#include <sstream>
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(TensorUtil)
{
    NNTestMethod(forEach)
    {
        NNTestParams(std::function, Tensor &)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4 });
            forEach([](T &t)
            {
                t *= 3.14;
            }, t);
            for(size_t i = 0; i < t.size(); ++i)
                NNTestAlmostEquals(t(i), i * 3.14, 1e-12);
            Tensor<T> almostTooBig(Storage<size_t>(NN_MAX_NUM_DIMENSIONS -1, 1), true);
            try
            {
                forEach([](T &){}, almostTooBig);
            }
            catch(const Error &)
            {
                NNTest(false);
            }
            Tensor<T> tooBig(Storage<size_t>(NN_MAX_NUM_DIMENSIONS, 1), true);
            try
            {
                forEach([](T &){}, tooBig);
                NNTest(false);
            }
            catch(const Error &)
            {}
        }
    }
}
