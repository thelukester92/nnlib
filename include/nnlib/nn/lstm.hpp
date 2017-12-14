#ifndef NN_LSTM_HPP
#define NN_LSTM_HPP

#include "linear.hpp"
#include "logistic.hpp"
#include "tanh.hpp"

namespace nnlib
{

template <typename T>
class LSTM;

template <typename T>
void swap(LSTM<T> &, LSTM<T> &);

/// \brief Long short-term memory recurrent module.
///
/// This implementation makes a strong assumption that inputs and outputs are matrices.
template <typename T = NN_REAL_T>
class LSTM : public Module<T>
{
public:
	LSTM(size_t inps, size_t outs);
	LSTM(const LSTM &module);
	LSTM(const Serialized &node);
	
	virtual ~LSTM();
	
	LSTM &operator=(LSTM module);
	
	friend void swap <> (LSTM &a, LSTM &b);
	
	LSTM &gradClip(T clip);
	T gradClip() const;
	
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
	Module<T> *m_inpGateX;
	Module<T> *m_inpGateY;
	Module<T> *m_inpGateH;
	Module<T> *m_inpGate;
	Module<T> *m_fgtGateX;
	Module<T> *m_fgtGateY;
	Module<T> *m_fgtGateH;
	Module<T> *m_fgtGate;
	Module<T> *m_inpModX;
	Module<T> *m_inpModY;
	Module<T> *m_inpMod;
	Module<T> *m_outGateX;
	Module<T> *m_outGateY;
	Module<T> *m_outGateH;
	Module<T> *m_outGate;
	Module<T> *m_outMod;
	
	Tensor<T> m_inpAdd;
	Tensor<T> m_fgtAdd;
	Tensor<T> m_outGrad;
	
	Tensor<T> m_state;
	Tensor<T> m_prevState;
	Tensor<T> m_prevOutput;
	Tensor<T> m_stateGrad;
	Tensor<T> m_curStateGrad;
	Tensor<T> m_gradBuffer;
	
	T m_clip;
	size_t m_outs;
};

}

NNRegisterType(LSTM, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::LSTM<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/lstm.tpp"
#endif

#endif
