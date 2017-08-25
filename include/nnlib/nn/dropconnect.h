#ifndef DROPCONNECT_H
#define DROPCONNECT_H

#include "module.h"
#include "linear.h"

namespace nnlib
{

template <typename T = double>
class DropConnect : public Container<T>
{
	using Container<T>::m_training;
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	
	DropConnect(T dropProbability = 0.1, size_t inps = 0, size_t outs = 0, size_t bats = 1) :
		m_module(new Linear<T>(inps, outs, bats)),
		m_params(&m_module->parameters()),
		m_backup(m_params->copy()),
		m_mask(m_params->size()),
		m_dropProbability(dropProbability)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
		Container<T>::add(m_module);
	}
	
	DropConnect(Module<T> *module, T dropProbability = 0.1) :
		m_module(module),
		m_params(module == nullptr ? nullptr : &m_module->parameters()),
		m_backup(module == nullptr ? Tensor<T>(0) : m_params->copy()),
		m_mask(module == nullptr ? 0 : m_params->size()),
		m_dropProbability(dropProbability)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
		Container<T>::add(m_module);
	}
	
	DropConnect(const DropConnect &module) :
		m_module(module.m_module == nullptr ? nullptr : module.m_module->copy()),
		m_params(module.m_module == nullptr ? nullptr : &module.m_module->parameters()),
		m_backup(module.m_module == nullptr ? Tensor<T>(0) : m_params->copy()),
		m_mask(module.m_module == nullptr ? 0 : m_params->size()),
		m_dropProbability(module.m_dropProbability)
	{
		m_training = module.m_training;
		Container<T>::add(m_module);
	}
	
	DropConnect &operator=(const DropConnect &module)
	{
		m_module			= module.m_module == nullptr ? nullptr : module.m_module->copy();
		m_params			= module.m_module == nullptr ? nullptr : &module.m_module->parameters();
		m_backup			= module.m_module == nullptr ? Tensor<T>(0) : Tensor<T>(m_params->size());
		m_mask				= module.m_mask.copy();
		m_dropProbability	= module.m_dropProbability;
		m_training			= module.m_training;
		Container<T>::clear();
		Container<T>::add(m_module);
		return *this;
	}
	
	/// Get the probability that an output is not dropped.
	T dropProbability() const
	{
		return m_dropProbability;
	}
	
	/// Set the probability that an output is not dropped.
	DropConnect &dropProbability(T dropProbability)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
		m_dropProbability = dropProbability;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual DropConnect &add(Module<T> *) override
	{
		throw Error("Cannot add components to a DropConnect module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Module<T> *remove(size_t) override
	{
		throw Error("Cannot remove components from a DropConnect module!");
	}
	
	/// Cannot remove a component from this container.
	virtual DropConnect &clear() override
	{
		throw Error("Cannot remove components from a DropConnect module!");
	}
	
	// MARK: DropConnect methods
	
	/// Get the module used by this DropConnect.
	Module<T> &module()
	{
		return *m_module;
	}
	
	/// \brief Set the module used by this DropConnect.
	///
	/// This also deletes the module previously used by this DropConnect.
	DropConnect &module(Module<T> &module)
	{
		m_module = &module;
		m_params = &module.parameters();
		m_backup = m_params->copy();
		m_mask.resize(m_params->size());
		Container<T>::clear();
		Container<T>::add(m_module);
		return *this;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		if(m_training)
		{
			m_backup.copy(*m_params);
			m_params->pointwiseProduct(m_mask.bernoulli(1 - m_dropProbability));
			return m_module->forward(input);
		}
		else
			return m_module->forward(input).scale(1 - m_dropProbability);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		if(m_training)
		{
			m_module->backward(input, outGrad);
			m_params->copy(m_backup);
			return m_module->inGrad();
		}
		else
			return m_module->backward(input, outGrad).scale(1 - m_dropProbability);
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_module->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_module->inGrad();
	}
	
	/// Set the input shape of this module, including batch.
	virtual DropConnect &inputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 2, "Expected matrix input!");
		m_module->inputs(dims);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	virtual DropConnect &outputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 2, "Expected matrix output!");
		m_module->outputs(dims);
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Container<T>::stateList();
		states.push_back(&m_mask);
		states.push_back(&m_backup);
		return states;
	}
	
	/// Save to a serialized node.
	virtual void save(SerializedNode &node) const override
	{
		node.set("module", m_module);
		node.set("dropProbability", m_dropProbability);
		node.set("training", m_training);
	}
	
	/// Load from a serialized node.
	virtual void load(const SerializedNode &node) override
	{
		module(*node.get<Module<T> *>("module"));
		node.get("dropProbability", m_dropProbability);
		node.get("training", m_training);
	}
	
	/*
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(m_module, m_dropProbability);
	}
	
	/// \brief Read from an archive.
	///
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		Module<T> *m;
		ar(m, m_dropProbability);
		module(*m);
	}
	*/
	
private:
	Module<T> *m_module;	///< The decorated module.
	Tensor<T> *m_params;	///< A pointer to the flattened parameters from m_module.
	Tensor<T> m_backup;		///< Backup buffer for the unmasked parameters.
	Tensor<T> m_mask;		///< Randomly-generated mask.
	T m_dropProbability;	///< The probability that an output is dropped.
};

}

NNRegisterType(DropConnect<float>, Module<float>);
NNRegisterType(DropConnect<double>, Module<double>);

#endif
