#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../core/tensor.h"
#include "../serialization/factory.h"
#include "../serialization/serialized.h"

namespace nnlib
{

/// \brief The abtract base class for all neural network modules.
///
/// \note The assignment operator invalidates parameters, grad, and state.
/// They must be reflattened to use them.
template <typename T = double>
class Module
{
public:
	Module() :
		m_training(true),
		m_sharedParamCount(0),
		m_sharedGradCount(0),
		m_sharedStateCount(0)
	{}
	
	Module(const Storage<size_t> &inps, const Storage<size_t> &outs) :
		m_outputShape(outs),
		m_inputShape(inps),
		m_training(true),
		m_sharedParamCount(0),
		m_sharedGradCount(0),
		m_sharedStateCount(0)
	{}
	
	Module(const Module &) = delete;
	Module &operator=(const Module &) = delete;
	virtual ~Module() {}
	
	// MARK: Serialization
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const = 0;
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) = 0;
	
	// MARK: Computation
	
	/// Forward propagate input, returning output.
	virtual const Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual const Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;
	
	// MARK: Size Management
	
	/// Change the output size, not including batch dimensions.
	virtual void resizeOutputs(const Storage<size_t> &dims)
	{
		m_outputShape = dims;
	}
	
	/// Change the input size, not including batch dimensions.
	virtual void resizeInputs(const Storage<size_t> &dims)
	{
		m_inputShape = dims;
	}
	
	/// Change the input and output sizes, not including batch dimensions.
	virtual void resize(const Storage<size_t> &inps, const Storage<size_t> &outs)
	{
		resizeOutputs(outs);
		resizeInputs(inps);
	}
	
	/// Change the output size, not including batch dimensions (convenience method).
	template <typename ... Ts>
	void resizeOutputs(Ts... dims)
	{
		resizeOutputs({ static_cast<size_t>(dims)... });
	}
	
	/// Change the input size, not including batch dimensions (convenience method).
	template <typename ... Ts>
	void resizeInputs(Ts... dims)
	{
		resizeInputs({ static_cast<size_t>(dims)... });
	}
	
	// MARK: Getters, etc.
	
	/// Get cached output.
	const Tensor<T> &output() const
	{
		return m_output;
	}
	
	/// Get cached input gradient.
	const Tensor<T> &inGrad() const
	{
		return m_inGrad;
	}
	
	/// Get output shape. This does not include batch or batch-like dimensions.
	const Storage<size_t> &outputShape() const
	{
		return m_outputShape;
	}
	
	/// Get input shape. This does not include batch or batch-like dimensions.
	const Storage<size_t> &inputShape() const
	{
		return m_inputShape;
	}
	
	/// /brief A convenience method for contructing a copy of the current module.
	///
	/// Rather than overriding this in a subclass, simply implement the copy constructor.
	/// This will only work if the actual type of the instance has been registered with NNRegisterType.
	virtual Module *copy() final
	{
		return Factory<Module>::constructCopy(this);
	}
	
	/// Returns whether this module is in training mode.
	virtual bool training() const
	{
		return m_training;
	}
	
	/// Sets whether this module is in training mode.
	virtual Module &training(bool training)
	{
		m_training = training;
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameterList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	/// By default, this is only the calculated output.
	virtual Storage<Tensor<T> *> stateList()
	{
		return { &output() };
	}
	
	/// Reset the internal state of this module.
	virtual Module &forget()
	{
		for(Tensor<T> *t : stateList())
			t->fill(0);
		return *this;
	}
	
	/// A flattened tensor of all of this module's parameters.
	Tensor<T> &parameters()
	{
		Storage<Tensor<T> *> list = parameterList();
		bool recalculate = m_sharedParamCount != m_flatParameters.sharedCount();
		
		for(size_t i = 0, count = list.size(); i != count && !recalculate; ++i)
		{
			if(!m_flatParameters.sharedWith(*list[i]))
				recalculate = true;
		}
		
		if(recalculate)
		{
			m_flatParameters = Tensor<T>::flatten(parameterList());
			m_sharedParamCount = m_flatParameters.sharedCount();
		}
		
		return m_flatParameters;
	}
	
	/// A flattened tensor of all of this module's parameters' gradients.
	Tensor<T> &grad()
	{
		Storage<Tensor<T> *> list = gradList();
		bool recalculate = m_sharedGradCount != m_flatGrad.sharedCount();
		
		for(size_t i = 0, count = list.size(); i != count && !recalculate; ++i)
		{
			if(!m_flatGrad.sharedWith(*list[i]))
				recalculate = true;
		}
		
		if(recalculate)
		{
			m_flatGrad = Tensor<T>::flatten(gradList());
			m_sharedGradCount = m_flatGrad.sharedCount();
		}
		
		return m_flatGrad;
	}
	
	/// A flattened tensor of all of this module's internal states.
	Tensor<T> &state()
	{
		Storage<Tensor<T> *> list = stateList();
		bool recalculate = m_sharedStateCount != m_flatState.sharedCount();
		
		for(size_t i = 0, count = list.size(); i != count && !recalculate; ++i)
		{
			if(!m_flatState.sharedWith(*list[i]))
				recalculate = true;
		}
		
		if(recalculate)
		{
			m_flatState = Tensor<T>::flatten(stateList());
			m_sharedStateCount = m_flatState.sharedCount();
		}
		
		return m_flatState;
	}
	
	/// Invalidate the cached flattened tensors.
	Module &invalidateCache()
	{
		m_sharedParamCount = 0;
		m_sharedGradCount = 0;
		m_sharedStateCount = 0;
	}
	
protected:
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	Storage<size_t> m_outputShape;
	Storage<size_t> m_inputShape;
	
	Tensor<T> m_flatParameters;
	Tensor<T> m_flatGrad;
	Tensor<T> m_flatState;
	bool m_training;
	
private:
	size_t m_sharedParamCount;
	size_t m_sharedGradCount;
	size_t m_sharedStateCount;
};

}

#endif
