#ifndef CORE_TENSOR_UTIL_TPP
#define CORE_TENSOR_UTIL_TPP

#include "tensor_util.hpp"

#ifndef NN_MAX_NUM_DIMENSIONS
#define NN_MAX_NUM_DIMENSIONS 32ul
#endif

namespace nnlib
{

namespace detail
{
    template <size_t D, size_t I>
    struct ForEachHelper
    {
        template <typename F, typename ... Ts>
        static void apply(Storage<size_t> &indices, const Storage<size_t> &shape, F func, Ts &...ts)
        {
            for(indices[I] = 0; indices[I] < shape[I]; ++indices[I])
                ForEachHelper<D-1, I+1>::apply(indices, shape, func, ts...);
        }
    };

    template <size_t I>
    struct ForEachHelper<1ul, I>
    {
        template <typename F, typename ... Ts>
        static void apply(Storage<size_t> &indices, const Storage<size_t> &shape, F func, Ts &...ts)
        {
            for(indices[I] = 0; indices[I] < shape[I]; ++indices[I])
                func(ts.ptr()[indexOf(ts, indices)]...);
        }

    private:
        template <typename T>
        static size_t indexOf(T &&tensor, const Storage<size_t> &indices)
        {
            NNAssertEquals(tensor.dims(), indices.size(), "Incompatible tensors in forEach!");
            const Storage<size_t> &strides = tensor.strides();
            size_t i = 0;
            for(size_t j = 0; j < I + 1; ++j)
            {
                NNAssertLessThan(indices[j], tensor.size(j), "Incompatible tensors in forEach!");
                i += indices[j] * strides[j];
            }
            return i;
        }
    };

    template <size_t D>
    struct ForEach
    {
        template <typename F, typename ... Ts>
        static void apply(const Storage<size_t> &shape, F func, Ts &...ts)
        {
            Storage<size_t> indices(D);
            ForEachHelper<D, 0>::apply(indices, shape, func, ts...);
        }
    };

    template <size_t MIN, size_t MAX, template <size_t> class WORKER>
    struct TemplateSearch
    {
        template <typename ... Ts>
        static void apply(size_t v, Ts &&...ts)
        {
            if(v == MIN)
                WORKER<MIN>::apply(std::forward<Ts>(ts)...);
            else
                TemplateSearch<MIN+1, MAX, WORKER>::apply(v, std::forward<Ts>(ts)...);
        }
    };

    template <size_t MAX, template <size_t> class WORKER>
    struct TemplateSearch<MAX, MAX, WORKER>
    {
        template <typename ... Ts>
        static void apply(size_t v, Ts &&...ts)
        {
            NNHardAssert(v == MAX, "Too many dimensions! Define NN_MAX_NUM_DIMENSIONS to increase this limit.");
            WORKER<MAX>::apply(std::forward<Ts>(ts)...);
        }
    };
}

/// A more efficient way apply a function to each element in one or more tensors.
template <typename F, typename T, typename ... Ts>
void forEach(F func, T &first, Ts &...ts)
{
    detail::TemplateSearch<1ul, NN_MAX_NUM_DIMENSIONS, detail::ForEach>::apply(first.dims(), first.shape(), func, first, ts...);
}

}

#endif
