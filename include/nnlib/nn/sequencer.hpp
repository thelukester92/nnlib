#ifndef NN_SEQUENCER_HPP
#define NN_SEQUENCER_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T>
class Sequencer;

template <typename T>
void swap(Sequencer<T> &, Sequencer<T> &);

/// \brief Allows an extra "sequence" dimension when passing in inputs to a module.
///
/// Inputs will be passed through to the inner module one at a time and backpropagated
/// in reverse order, essentially abstracting away BPTT.
template <typename T = NN_REAL_T>
class Sequencer : public Module<T>
{
public:
    Sequencer(Module<T> *module, bool reverse = false);
    Sequencer(const Sequencer &module);
    Sequencer(const Serialized &node);

    virtual ~Sequencer();

    Sequencer &operator=(Sequencer module);

    friend void swap <> (Sequencer &a, Sequencer &b);

    Module<T> &module();
    Sequencer &module(Module<T> *module);

    bool isReversed();
    Sequencer &reverse(bool reverse = true);

    /// Begin a sequence. This is automatically called by forward.
    void startForward(const Tensor<T> &first, size_t sequenceLength);

    /// Forward the next sample in the sequence. This is automatically called by forward.
    void stepForward(const Tensor<T> &singleInput, size_t i);

    /// Backward the next sample in the sequence. This is automatically called by backward.
    void stepBackward(const Tensor<T> &singleInput, const Tensor<T> &singleOutGrad, size_t i);

    virtual void training(bool training = true) override;
    virtual void forget() override;

    virtual void save(Serialized &node) const override;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;
    virtual Storage<Tensor<T> *> stateList() override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;

private:
    Module<T> *m_module;
    Tensor<T> m_states;
    bool m_reverse;
};

}

NNRegisterType(Sequencer, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Sequencer<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/sequencer.tpp"
#endif

#endif
