#include "../test_tensor_iterator.hpp"
#include "nnlib/core/detail/tensor_iterator.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(TensorIterator)
{
    NNTestMethod(TensorIterator)
    {
        NNTestParams(const Tensor *)
        {
            Tensor<T> t(5);
            TensorIterator<T> itr(&t);
            NNTestEquals(t.ptr(), &*itr);
            TensorIterator<const T> citr(&t);
            NNTestEquals(t.ptr(), &*citr);
        }

        NNTestParams(const Tensor *, bool)
        {
            Tensor<T> t(5);
            TensorIterator<T> itr(&t, true);
            NNTestEquals(t.ptr() + 5, &*itr);
            TensorIterator<const T> citr(&t, true);
            NNTestEquals(t.ptr() + 5, &*citr);
        }
    }

    NNTestMethod(operator++)
    {
        NNTestParams()
        {
            Tensor<T> t(2, 5);
            TensorIterator<T> itr(&t);
            TensorIterator<T> itr2 = ++itr;
            NNTestEquals(t.ptr() + 1, &*itr);
            NNTestEquals(t.ptr() + 1, &*itr2);
            Tensor<T> s = t.transpose();
            TensorIterator<T> itr3(&s);
            ++itr3;
            ++itr3;
            NNTestEquals(s.ptr() + 1, &*itr3);
        }

        NNTestParams(int)
        {
            Tensor<T> t(5);
            TensorIterator<T> itr(&t);
            TensorIterator<T> itr2 = itr++;
            NNTestEquals(t.ptr() + 1, &*itr);
            NNTestEquals(t.ptr(), &*itr2);
        }
    }

    NNTestMethod(operator*)
    {
        NNTestParams()
        {
            Tensor<T> t({ 1, 2, 3 });
            TensorIterator<T> itr(&t);
            NNTestEquals(*itr++, 1);
            NNTestEquals(*itr++, 2);
            NNTestEquals(*itr++, 3);
        }
    }

    NNTestMethod(operator==)
    {
        NNTestParams(const TensorIterator &)
        {
            Tensor<T> t(5, 3);
            TensorIterator<T> itr1(&t), itr2(&t);
            NNTest(itr1 == itr2);
            NNTest(!(++itr1 == itr2));
            Tensor<T> s = t.transpose();
            TensorIterator<T> itr3(&s), itr4(&s);
            NNTest(itr3 == itr4);
            NNTest(!(++itr3 == itr4));
        }
    }

    NNTestMethod(operator!=)
    {
        NNTestParams(const TensorIterator &)
        {
            Tensor<T> t(5, 3);
            TensorIterator<T> itr1(&t), itr2(&t);
            NNTest(!(itr1 != itr2));
            NNTest(++itr1 != itr2);
            Tensor<T> s = t.transpose();
            TensorIterator<T> itr3(&s), itr4(&s);
            NNTest(!(itr3 != itr4));
            NNTest(++itr3 != itr4);
        }
    }
}
