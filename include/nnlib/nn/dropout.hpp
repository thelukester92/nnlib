#ifndef NN_DROPOUT_HPP
#define NN_DROPOUT_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T = double>
class Dropout : public Module<T>
{
public:
	Dropout(T dropProbability = 0.1);
	Dropout(const Dropout &module);
	Dropout(const Serialized &node);
	Dropout &operator=(const Dropout &module);
	
	T dropProbability() const;
	Dropout &dropProbability(T dropProbability);
	
	bool isTraining() const;
	virtual void training(bool training = true) override;
	
	virtual void save(Serialized &node) const override;
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	
	virtual Storage<Tensor<T> *> stateList() override;
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Tensor<T> m_mask;
	T m_dropProbability;
	bool m_training;
};

}

NNRegisterType(Dropout, Module);
NNTemplateDefinition(Dropout, "detail/dropout.tpp");

#endif
