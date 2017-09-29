#ifndef NN_LSTM_H
#define NN_LSTM_H

#include "linear.h"
#include "logistic.h"
#include "tanh.h"

namespace nnlib
{

/// \brief Long short-term memory recurrent module.
///
/// This implementation makes a strong assumption that inputs and outputs are matrices.
template <typename T = double>
class LSTM : public Module<T>
{
public:
	LSTM(size_t inps, size_t outs) :
		m_inpGateX(new Linear<T>(inps, outs)),
		m_inpGateY(new Linear<T>(outs, outs)),
		m_inpGateH(new Linear<T>(outs, outs)),
		m_inpGate(new Logistic<T>()),
		m_fgtGateX(new Linear<T>(inps, outs)),
		m_fgtGateY(new Linear<T>(outs, outs)),
		m_fgtGateH(new Linear<T>(outs, outs)),
		m_fgtGate(new Logistic<T>()),
		m_inpModX(new Linear<T>(inps, outs)),
		m_inpModY(new Linear<T>(outs, outs)),
		m_inpMod(new TanH<T>()),
		m_outGateX(new Linear<T>(inps, outs)),
		m_outGateY(new Linear<T>(outs, outs)),
		m_outGateH(new Linear<T>(outs, outs)),
		m_outGate(new Logistic<T>()),
		m_outMod(new TanH<T>()),
		m_resetGrad(true),
		m_clip(0),
		m_outs(outs)
	{
		forget();
	}
	
	LSTM(const LSTM &module) :
		m_inpGateX(module.m_inpGateX->copy()),
		m_inpGateY(module.m_inpGateY->copy()),
		m_inpGateH(module.m_inpGateH->copy()),
		m_inpGate(module.m_inpGate->copy()),
		m_fgtGateX(module.m_fgtGateX->copy()),
		m_fgtGateY(module.m_fgtGateY->copy()),
		m_fgtGateH(module.m_fgtGateH->copy()),
		m_fgtGate(module.m_fgtGate->copy()),
		m_inpModX(module.m_inpModX->copy()),
		m_inpModY(module.m_inpModY->copy()),
		m_inpMod(module.m_inpMod->copy()),
		m_outGateX(module.m_outGateX->copy()),
		m_outGateY(module.m_outGateY->copy()),
		m_outGateH(module.m_outGateH->copy()),
		m_outGate(module.m_outGate->copy()),
		m_outMod(module.m_outMod->copy()),
		m_resetGrad(module.m_resetGrad),
		m_clip(module.m_clip),
		m_outs(module.m_outs)
	{}
	
	LSTM(const Serialized &node) :
		m_inpGateX(node.get<Module<T> *>("inpGateX")),
		m_inpGateY(node.get<Module<T> *>("inpGateY")),
		m_inpGateH(node.get<Module<T> *>("inpGateH")),
		m_inpGate(node.get<Module<T> *>("inpGate")),
		m_fgtGateX(node.get<Module<T> *>("fgtGateX")),
		m_fgtGateY(node.get<Module<T> *>("fgtGateY")),
		m_fgtGateH(node.get<Module<T> *>("fgtGateH")),
		m_fgtGate(node.get<Module<T> *>("fgtGate")),
		m_inpModX(node.get<Module<T> *>("inpModX")),
		m_inpModY(node.get<Module<T> *>("inpModY")),
		m_inpMod(node.get<Module<T> *>("inpMod")),
		m_outGateX(node.get<Module<T> *>("outGateX")),
		m_outGateY(node.get<Module<T> *>("outGateY")),
		m_outGateH(node.get<Module<T> *>("outGateH")),
		m_outGate(node.get<Module<T> *>("outGate")),
		m_outMod(node.get<Module<T> *>("outMod")),
		m_resetGrad(node.get<bool>("resetGrad")),
		m_clip(node.get<T>("clip")),
		m_outs(node.get<size_t>("outs"))
	{}
	
	virtual ~LSTM()
	{
		delete m_inpGateX;
		delete m_inpGateY;
		delete m_inpGateH;
		delete m_inpGate;
		delete m_fgtGateX;
		delete m_fgtGateY;
		delete m_fgtGateH;
		delete m_fgtGate;
		delete m_inpModX;
		delete m_inpModY;
		delete m_inpMod;
		delete m_outGateX;
		delete m_outGateY;
		delete m_outGateH;
		delete m_outGate;
		delete m_outMod;
	}
	
	LSTM &operator=(LSTM module)
	{
		swap(*this, module);
		return *this;
	}
	
	friend void swap(LSTM &a, LSTM &b)
	{
		using std::swap;
		swap(a.m_inpGateX, b.m_inpGateX);
		swap(a.m_inpGateY, b.m_inpGateY);
		swap(a.m_inpGateH, b.m_inpGateH);
		swap(a.m_inpGate, b.m_inpGate);
		swap(a.m_fgtGateX, b.m_fgtGateX);
		swap(a.m_fgtGateY, b.m_fgtGateY);
		swap(a.m_fgtGateH, b.m_fgtGateH);
		swap(a.m_fgtGate, b.m_fgtGate);
		swap(a.m_inpModX, b.m_inpModX);
		swap(a.m_inpModY, b.m_inpModY);
		swap(a.m_inpMod, b.m_inpMod);
		swap(a.m_outGateX, b.m_outGateX);
		swap(a.m_outGateY, b.m_outGateY);
		swap(a.m_outGateH, b.m_outGateH);
		swap(a.m_outGate, b.m_outGate);
		swap(a.m_outMod, b.m_outMod);
		swap(a.m_resetGrad, b.m_resetGrad);
		swap(a.m_clip, b.m_clip);
		swap(a.m_outs, b.m_outs);
	}
	
	LSTM &gradClip(T clip)
	{
		m_clip = clip;
		return *this;
	}
	
	T gradClip() const
	{
		return m_clip;
	}
	
	virtual void forget() override
	{
		Module<T>::forget();
		m_outMod->output().fill(0);
		m_resetGrad = true;
	}
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("inpGateX", m_inpGateX);
		node.set("inpGateY", m_inpGateY);
		node.set("inpGateH", m_inpGateH);
		node.set("inpGate", m_inpGate);
		node.set("fgtGateX", m_fgtGateX);
		node.set("fgtGateY", m_fgtGateY);
		node.set("fgtGateH", m_fgtGateH);
		node.set("fgtGate", m_fgtGate);
		node.set("inpModX", m_inpModX);
		node.set("inpModY", m_inpModY);
		node.set("inpMod", m_inpMod);
		node.set("outGateX", m_outGateX);
		node.set("outGateY", m_outGateY);
		node.set("outGateH", m_outGateH);
		node.set("outGate", m_outGate);
		node.set("outMod", m_outMod);
		node.set("resetGrad", m_resetGrad);
		node.set("clip", m_clip);
		node.set("outs", m_outs);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_state.resize(input.size(0), m_outs);
		m_prevState.resize(input.size(0), m_outs);
		m_prevState.copy(m_state);
		
		m_outMod->output().resize(input.size(0), m_outs);
		m_prevOutput.resize(input.size(0), m_outs);
		m_prevOutput.copy(m_outMod->output());
		
		// input gate
		m_inpGateX->forward(input);
		m_inpGateX->output().addM(m_inpGateY->forward(m_prevOutput));
		m_inpGateX->output().addM(m_inpGateH->forward(m_prevState));
		m_inpGate->forward(m_inpGateX->output());
		
		// forget gate
		m_fgtGateX->forward(input);
		m_fgtGateX->output().addM(m_fgtGateY->forward(m_prevOutput));
		m_fgtGateX->output().addM(m_fgtGateH->forward(m_prevState));
		m_fgtGate->forward(m_fgtGateX->output());
		
		// input value
		m_inpModX->forward(input);
		m_inpModX->output().addM(m_inpModY->forward(m_prevOutput));
		m_inpMod->forward(m_inpModX->output());
		
		// update memory cell (hidden state)
		m_inpAdd.resize(m_inpGate->output().shape());
		m_inpAdd.copy(m_inpGate->output()).pointwiseProduct(m_inpMod->output());
		m_fgtAdd.resize(m_fgtGate->output().shape());
		m_fgtAdd.copy(m_fgtGate->output()).pointwiseProduct(m_state);
		m_state.copy(m_inpAdd).addM(m_fgtAdd);
		
		// output gate
		m_outGateX->forward(input);
		m_outGateX->output().addM(m_outGateY->forward(m_prevOutput));
		m_outGateX->output().addM(m_outGateH->forward(m_state));
		m_outGate->forward(m_outGateX->output());
		m_outAdd.resize(m_outGate->output().shape());
		m_outAdd.copy(m_outGate->output()).pointwiseProduct(m_state);
		
		// final output
		return m_output = m_outMod->forward(m_outAdd);
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
		m_outGrad.resize(input.size(0), m_outs);
		m_stateGrad.resize(input.size(0), m_outs);
		m_curStateGrad.resize(input.size(0), m_outs);
		m_gradBuffer.resize(input.size(0), m_outs);
		m_inGrad.resize(input.shape());
		
		if(m_resetGrad)
		{
			m_resetGrad = false;
			m_outGrad.fill(0);
			m_stateGrad.fill(0);
		}
		
		// update output gradient
		m_outGrad.addM(outGrad);
		m_outMod->backward(m_outAdd, m_outGrad);
		
		// backprop to hidden state
		m_curStateGrad.copy(m_outMod->inGrad()).pointwiseProduct(m_outGate->output());
		m_curStateGrad.addM(m_stateGrad);
		
		// backprop through output gate
		m_gradBuffer.copy(m_outMod->inGrad()).pointwiseProduct(m_state);
		m_outGate->backward(m_outGateX->output(), m_gradBuffer);
		m_inGrad.copy(m_outGateX->backward(input, m_outGate->inGrad()));
		m_outGrad.copy(m_outGateY->backward(m_prevOutput, m_outGate->inGrad()));
		
		// backprop through input value
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_inpGate->output());
		m_inpMod->backward(m_inpModX->output(), m_gradBuffer);
		m_inGrad.addM(m_inpModX->backward(input, m_inpMod->inGrad()));
		m_outGrad.addM(m_inpModY->backward(m_prevOutput, m_inpMod->inGrad()));
		
		// backprop through forget gate
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_prevState);
		m_fgtGate->backward(m_fgtGateX->output(), m_gradBuffer);
		m_inGrad.addM(m_fgtGateX->backward(input, m_fgtGate->inGrad()));
		m_stateGrad.copy(m_fgtGateH->backward(m_prevState, m_fgtGate->inGrad()));
		m_outGrad.addM(m_fgtGateY->backward(m_prevOutput, m_fgtGate->inGrad()));
		
		// backprop through input gate
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_inpMod->output());
		m_inpGate->backward(m_inpGateX->output(), m_gradBuffer);
		m_inGrad.addM(m_inpGateX->backward(input, m_inpGate->inGrad()));
		m_stateGrad.addM(m_inpGateH->backward(m_prevState, m_inpGate->inGrad()));
		m_outGrad.addM(m_inpGateY->backward(m_prevOutput, m_inpGate->inGrad()));
		
		// backprop to hidden state
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_fgtGate->output());
		m_stateGrad.addM(m_gradBuffer);
		
		// clip if necessary
		if(m_clip != 0)
			m_inGrad.clip(-m_clip, m_clip);
		
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		Storage<Tensor<T> *> list;
		list.append(m_inpGateX->paramsList());
		list.append(m_inpGateY->paramsList());
		list.append(m_inpGateH->paramsList());
		list.append(m_inpGate->paramsList());
		list.append(m_fgtGateX->paramsList());
		list.append(m_fgtGateY->paramsList());
		list.append(m_fgtGateH->paramsList());
		list.append(m_fgtGate->paramsList());
		list.append(m_inpModX->paramsList());
		list.append(m_inpModY->paramsList());
		list.append(m_inpMod->paramsList());
		list.append(m_outGateX->paramsList());
		list.append(m_outGateY->paramsList());
		list.append(m_outGateH->paramsList());
		list.append(m_outGate->paramsList());
		list.append(m_outMod->paramsList());
		return list;
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		Storage<Tensor<T> *> list;
		list.append(m_inpGateX->gradList());
		list.append(m_inpGateY->gradList());
		list.append(m_inpGateH->gradList());
		list.append(m_inpGate->gradList());
		list.append(m_fgtGateX->gradList());
		list.append(m_fgtGateY->gradList());
		list.append(m_fgtGateH->gradList());
		list.append(m_fgtGate->gradList());
		list.append(m_inpModX->gradList());
		list.append(m_inpModY->gradList());
		list.append(m_inpMod->gradList());
		list.append(m_outGateX->gradList());
		list.append(m_outGateY->gradList());
		list.append(m_outGateH->gradList());
		list.append(m_outGate->gradList());
		list.append(m_outMod->gradList());
		return list;
	}
	
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> list;
		list.append(m_inpGateX->stateList());
		list.append(m_inpGateY->stateList());
		list.append(m_inpGateH->stateList());
		list.append(m_inpGate->stateList());
		list.append(m_fgtGateX->stateList());
		list.append(m_fgtGateY->stateList());
		list.append(m_fgtGateH->stateList());
		list.append(m_fgtGate->stateList());
		list.append(m_inpModX->stateList());
		list.append(m_inpModY->stateList());
		list.append(m_inpMod->stateList());
		list.append(m_outGateX->stateList());
		list.append(m_outGateY->stateList());
		list.append(m_outGateH->stateList());
		list.append(m_outGate->stateList());
		list.append(m_outMod->stateList());
		return list.append({ &m_state, &m_prevState, &m_prevOutput, &m_outAdd });
	}
	
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
	Tensor<T> m_outAdd;
	Tensor<T> m_outGrad;
	
	Tensor<T> m_state;
	Tensor<T> m_prevState;
	Tensor<T> m_prevOutput;
	Tensor<T> m_stateGrad;
	Tensor<T> m_curStateGrad;
	Tensor<T> m_gradBuffer;
	
	bool m_resetGrad;
	T m_clip;
	size_t m_outs;
};

}

NNRegisterType(LSTM, Module);

#endif
