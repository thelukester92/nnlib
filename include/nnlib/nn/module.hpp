#ifndef NN_MODULE_HPP
#define NN_MODULE_HPP

#include "../core/tensor.hpp"
#include "../serialization/factory.hpp"
#include "../serialization/serialized.hpp"

namespace nnlib
{

/// The abtract base class for all neural network modules.
template <typename T = double>
class Module
{
public:
	/// Explicitly default the no-argument constructor.
	Module() = default;
	
	/// No copy construction possible at this level.
	Module(const Module &) = delete;
	
	/// Virtualize the destructor.
	virtual ~Module() = default;
	
	/// No assignment possible at this level.
	Module &operator=(const Module &) = delete;
	
	/// Construct a copy of this module. Must be registered with NNRegisterType.
	Module *copy() const
	{
		return Factory<Module>::constructCopy(this);
	}
	
	/// Set whether this module is in training mode. Useful for modules like batchnorm that behave differently at evaluation time.
	virtual void training(bool training = true) {}
	
	/// Reset the internal state of this module. Useful for recurrent modules that have additional inner state.
	virtual void forget()
	{
		state().fill(0);
	}
	
	// MARK: Serialization
	
	/// \brief Save the current module to a serialized node.
	///
	/// The load method is omitted; instead, a constructor taking a Serialized& should be implemented
	/// in subclasses of Module.
	virtual void save(Serialized &) const = 0;
	
	// MARK: Computation
	
	/// Evaluate the module and return the new output.
	virtual Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Take the derivative of the module and return the gradient of the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList()
	{
		return {};
	}
	
	virtual Storage<Tensor<T> *> gradList()
	{
		return {};
	}
	
	virtual Storage<Tensor<T> *> stateList()
	{
		return { &m_output };
	}
	
	Tensor<T> &params()
	{
		auto list = paramsList();
		if(!m_params.sharedWith(list))
			m_params = Tensor<T>::vectorize(list);
		return m_params;
	}
	
	Tensor<T> &grad()
	{
		auto list = gradList();
		if(!m_grad.sharedWith(list))
			m_grad = Tensor<T>::vectorize(list);
		return m_grad;
	}
	
	Tensor<T> &state()
	{
		auto list = stateList();
		if(!m_state.sharedWith(list))
			m_state = Tensor<T>::vectorize(list);
		return m_state;
	}
	
	// MARK: Getters
	
	virtual Tensor<T> &output()
	{
		return m_output;
	}
	
	const Tensor<T> &output() const
	{
		return const_cast<Tensor<T> *>(this)->output();
	}
	
	virtual Tensor<T> &inGrad()
	{
		return m_inGrad;
	}
	
	const Tensor<T> &inGrad() const
	{
		return const_cast<Tensor<T> *>(this)->inGrad();
	}
	
protected:
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	
	Tensor<T> m_params;
	Tensor<T> m_grad;
	Tensor<T> m_state;
};

}

#endif
