#ifndef NN_BATCHNORM_HPP
#define NN_BATCHNORM_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T>
class BatchNorm;

template <typename T>
void swap(BatchNorm<T> &, BatchNorm<T> &);

/// \brief Batch normalization.
///
/// Works only on 2D tensors.
template <typename T = NN_REAL_T>
class BatchNorm : public Module<T>
{
public:
    BatchNorm(size_t inps);
    BatchNorm(const BatchNorm &module);
    BatchNorm(const Serialized &node);
    BatchNorm &operator=(BatchNorm module);
    
    friend void swap <> (BatchNorm &a, BatchNorm &b);
    
    BatchNorm &reset();
    
    Tensor<T> weights();
    Tensor<T> bias();
    
    T momentum() const;
    BatchNorm &momentum(T momentum);
    
    bool isTraining() const;
    virtual void training(bool training = true) override;
    
    // MARK: Serialization
    
    virtual void save(Serialized &node) const override;
    
    // MARK: Computation
    
    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
    
    // MARK: Buffers
    
    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;
    virtual Storage<Tensor<T> *> stateList() override;
    
protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;
    
private:
    Tensor<T> m_weights;		///< How much to scale the normalized input.
    Tensor<T> m_weightsGrad;	///< Gradient of error w.r.t. weights.
    Tensor<T> m_biases;			///< How much to shift the normalized input.
    Tensor<T> m_biasesGrad;		///< Gradient of error w.r.t. biases.
    
    Tensor<T> m_runningMeans;	///< A running mean for each input, for after training.
    Tensor<T> m_runningVars;	///< A running variance for each input, for after training.
    
    Tensor<T> m_means;			///< Mean of each input dimension within the batch.
    Tensor<T> m_invStds;		///< Inverted standard deviation of each input dimension within the batch.
    
    T m_momentum;				///< How much to update running mean and variance.
    bool m_training;			///< Whether this module is in training or evaluation mode.
};

}

NNRegisterType(BatchNorm, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::BatchNorm<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/batchnorm.tpp"
#endif

#endif
