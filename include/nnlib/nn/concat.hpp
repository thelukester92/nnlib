#ifndef NN_CONCAT_HPP
#define NN_CONCAT_HPP

#include "container.hpp"

namespace nnlib
{

/// \brief Feed in one input to a set of modules and concatenate output.
///
/// In order to avoid assumptions about module output dimensions,
/// the concatenation dimension is a parameter.
/// For example, consider two modules with the following outputs:
///     | a b c |    | j k l |
///     | d e f |    | m n o |
///     | g h i |    | p q r |.
/// If the concatenation dimension is 0, the result will be
///     | a b c |
///     | d e f |
///     | g h i |
///     | j k l |
///     | m n o |
///     | p q r |.
/// If the concatenation dimension is 1, the result will be
///     | a b c j k l |
///     | d e f m n o |
///     | g h i p q r |.
/// Modules may not produce square outputs, so often only one dimension will actually work.
/// By default, the last dimension will be used at the concatenation dimension.
template <typename T = NN_REAL_T>
class Concat : public Container<T>
{
public:
    using Container<T>::add;
    using Container<T>::components;

    template <typename ... Ms>
    Concat(Module<T> *first, Ms... rest) :
        Container<T>(first->inputShape(), Tensor<T>::concatenate({ &first->output(), &rest->output()... }, (size_t) -1).shape(), first, rest...),
        m_concatDim(first->outputShape().size() - 1)
    {}

    Concat(const Concat &module);
    Concat(const Serialized &node);

    Concat &operator=(const Concat &module);

    size_t concatDim() const;
    Concat &concatDim(size_t dim);

    virtual void save(Serialized &node) const override;

    virtual Concat &add(Module<T> *component) override;
    virtual Module<T> *remove(size_t index) override;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;
    using Container<T>::m_components;

private:
    size_t m_concatDim;
};

}

NNRegisterType(Concat, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Concat<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/concat.tpp"
#endif

#endif
