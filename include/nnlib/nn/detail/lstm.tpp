#ifndef NN_LSTM_TPP
#define NN_LSTM_TPP

#include "../lstm.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
LSTM<T>::LSTM(size_t inps, size_t outs) :
	m_inpGateX(new Linear<T>(inps, outs, false)),
	m_inpGateY(new Linear<T>(outs, outs, false)),
	m_inpGateH(new Linear<T>(outs, outs)),
	m_inpGate(new Logistic<T>()),
	m_fgtGateX(new Linear<T>(inps, outs, false)),
	m_fgtGateY(new Linear<T>(outs, outs, false)),
	m_fgtGateH(new Linear<T>(outs, outs)),
	m_fgtGate(new Logistic<T>()),
	m_inpModX(new Linear<T>(inps, outs, false)),
	m_inpModY(new Linear<T>(outs, outs)),
	m_inpMod(new TanH<T>()),
	m_outGateX(new Linear<T>(inps, outs, false)),
	m_outGateY(new Linear<T>(outs, outs, false)),
	m_outGateH(new Linear<T>(outs, outs)),
	m_outGate(new Logistic<T>()),
	m_outMod(new TanH<T>()),
	m_clip(0),
	m_outs(outs)
{
	forget();
}

template <typename T>
LSTM<T>::LSTM(const LSTM<T> &module) :
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
	m_clip(module.m_clip),
	m_outs(module.m_outs)
{}

template <typename T>
LSTM<T>::LSTM(const Serialized &node) :
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
	m_clip(node.get<T>("clip")),
	m_outs(node.get<size_t>("outs"))
{}

template <typename T>
LSTM<T>::~LSTM()
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

template <typename T>
LSTM<T> &LSTM<T>::operator=(LSTM<T> module)
{
	swap(*this, module);
	return *this;
}

template <typename T>
void swap(LSTM<T> &a, LSTM<T> &b)
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
	swap(a.m_clip, b.m_clip);
	swap(a.m_outs, b.m_outs);
}

template <typename T>
LSTM<T> &LSTM<T>::gradClip(T clip)
{
	m_clip = clip;
	return *this;
}

template <typename T>
T LSTM<T>::gradClip() const
{
	return m_clip;
}

template <typename T>
void LSTM<T>::forget()
{
	Module<T>::forget();
	m_output.fill(0);
	m_outGrad.fill(0);
	m_stateGrad.fill(0);
}

template <typename T>
void LSTM<T>::save(Serialized &node) const
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
	node.set("clip", m_clip);
	node.set("outs", m_outs);
}

template <typename T>
Tensor<T> &LSTM<T>::forward(const Tensor<T> &input)
{
	m_state.resize(input.size(0), m_outs);
	m_prevState.resize(input.size(0), m_outs);
	m_prevState.copy(m_state);

	m_output.resize(input.size(0), m_outs);
	m_prevOutput.resize(input.size(0), m_outs);
	m_prevOutput.copy(m_output);

	// input gate
	m_inpGateX->forward(input);
	Algebra<T>::mAdd_m(m_inpGateY->forward(m_prevOutput), m_inpGateX->output());
	Algebra<T>::mAdd_m(m_inpGateH->forward(m_prevState), m_inpGateX->output());
	m_inpGate->forward(m_inpGateX->output());

	// forget gate
	m_fgtGateX->forward(input);
	Algebra<T>::mAdd_m(m_fgtGateY->forward(m_prevOutput), m_fgtGateX->output());
	Algebra<T>::mAdd_m(m_fgtGateH->forward(m_prevState), m_fgtGateX->output());
	m_fgtGate->forward(m_fgtGateX->output());

	// input value
	m_inpModX->forward(input);
	Algebra<T>::mAdd_m(m_inpModY->forward(m_prevOutput), m_inpModX->output());
	m_inpMod->forward(m_inpModX->output());

	// update memory cell (hidden state)
	m_inpAdd.resize(m_inpGate->output().shape());
	m_fgtAdd.resize(m_fgtGate->output().shape());
	math::pointwiseProduct(m_inpGate->output(), m_inpMod->output(), m_inpAdd);
	math::pointwiseProduct(m_fgtGate->output(), m_prevState, m_fgtAdd);
	Algebra<T>::mAdd_m(m_inpAdd, m_state, 1, 0);
	Algebra<T>::mAdd_m(m_fgtAdd, m_state);
	m_outMod->forward(m_state);

	// output gate
	m_outGateX->forward(input);
	Algebra<T>::mAdd_m(m_outGateY->forward(m_prevOutput), m_outGateX->output());
	Algebra<T>::mAdd_m(m_outGateH->forward(m_state), m_outGateX->output());
	m_outGate->forward(m_outGateX->output());

	// final output
	return math::pointwiseProduct(m_outGate->output(), m_outMod->output(), m_output);
}

template <typename T>
Tensor<T> &LSTM<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
	NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
	m_outGrad.resize(input.size(0), m_outs);
	m_stateGrad.resize(input.size(0), m_outs);
	m_curStateGrad.resize(input.size(0), m_outs);
	m_gradBuffer.resize(input.size(0), m_outs);
	m_inGrad.resize(input.shape());

	// update output gradient
	Algebra<T>::mAdd_m(outGrad, m_outGrad);

	// backprop to hidden state
	math::pointwiseProduct(m_outGrad, m_outGate->output(), m_curStateGrad);
	m_curStateGrad.copy(m_outMod->backward(m_state, m_curStateGrad));
	Algebra<T>::mAdd_m(m_stateGrad, m_curStateGrad);

	// backprop through output gate
	math::pointwiseProduct(m_outGrad, m_outMod->output(), m_gradBuffer);
	m_outGate->backward(m_outGateX->output(), m_gradBuffer);
	m_inGrad.copy(m_outGateX->backward(input, m_outGate->inGrad()));
	m_outGrad.copy(m_outGateY->backward(m_prevOutput, m_outGate->inGrad()));
	Algebra<T>::mAdd_m(m_outGateH->backward(m_state, m_outGate->inGrad()), m_curStateGrad);

	// backprop through input value
	math::pointwiseProduct(m_curStateGrad, m_inpGate->output(), m_gradBuffer);
	m_inpMod->backward(m_inpModX->output(), m_gradBuffer);
	Algebra<T>::mAdd_m(m_inpModX->backward(input, m_inpMod->inGrad()), m_inGrad);
	Algebra<T>::mAdd_m(m_inpModY->backward(m_prevOutput, m_inpMod->inGrad()), m_outGrad);

	// backprop through forget gate
	math::pointwiseProduct(m_curStateGrad, m_prevState, m_gradBuffer);
	m_fgtGate->backward(m_fgtGateX->output(), m_gradBuffer);
	m_stateGrad.copy(m_fgtGateH->backward(m_prevState, m_fgtGate->inGrad()));
	Algebra<T>::mAdd_m(m_fgtGateX->backward(input, m_fgtGate->inGrad()), m_inGrad);
	Algebra<T>::mAdd_m(m_fgtGateY->backward(m_prevOutput, m_fgtGate->inGrad()), m_outGrad);

	// backprop through input gate
	math::pointwiseProduct(m_curStateGrad, m_inpMod->output(), m_gradBuffer);
	m_inpGate->backward(m_inpGateX->output(), m_gradBuffer);
	Algebra<T>::mAdd_m(m_inpGateX->backward(input, m_inpGate->inGrad()), m_inGrad);
	Algebra<T>::mAdd_m(m_inpGateH->backward(m_prevState, m_inpGate->inGrad()), m_stateGrad);
	Algebra<T>::mAdd_m(m_inpGateY->backward(m_prevOutput, m_inpGate->inGrad()), m_outGrad);

	// backprop to hidden state
	math::pointwiseProduct(m_curStateGrad, m_fgtGate->output(), m_gradBuffer);
	Algebra<T>::mAdd_m(m_gradBuffer, m_stateGrad);

	// clip if necessary
	if(m_clip != 0)
		math::clip(m_inGrad, -m_clip, m_clip);

	return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> LSTM<T>::paramsList()
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

template <typename T>
Storage<Tensor<T> *> LSTM<T>::gradList()
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

template <typename T>
Storage<Tensor<T> *> LSTM<T>::stateList()
{
	Storage<Tensor<T> *> list = Module<T>::stateList();
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
	return list.append({ &m_state, &m_prevState, &m_prevOutput });
}

}

#endif
