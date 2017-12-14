#ifndef DROPCONNECT_HPP
#define DROPCONNECT_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T>
class DropConnect;

template <typename T>
void swap(DropConnect<T> &, DropConnect<T> &);

/// A module decorator that randomly drops parameters with a given probability.
template <typename T = NN_REAL_T>
class DropConnect : public Module<T>
{
public:
	DropConnect(Module<T> *module, T dropProbability = 0.1);
	DropConnect(const DropConnect &module);
	DropConnect(const Serialized &node);
	
	DropConnect &operator=(DropConnect module);
	
	virtual ~DropConnect();
	
	friend void swap <> (DropConnect &a, DropConnect &b);
	
	/// Get the module this is decorating.
	Module<T> &module();
	
	/// Set the module this is decorating.
	DropConnect &module(Module<T> *module);
	
	/// Get the probability that an output is not dropped.
	T dropProbability() const;
	
	/// Set the probability that an output is not dropped.
	DropConnect &dropProbability(T dropProbability);
	
	bool isTraining() const;
	
	virtual void training(bool training = true) override;
	virtual void forget() override;
	
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
	Module<T> *m_module;
	Tensor<T> m_backup;
	Tensor<T> m_mask;
	T m_dropProbability;
	bool m_training;
};

}

NNRegisterType(DropConnect, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::DropConnect<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/dropconnect.tpp"
#endif

#endif
