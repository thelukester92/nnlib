#ifndef LSTM_H
#define LSTM_H

#include "../module.h"
#include "../util/random.h"
#include "../activations/logistic.h"
#include "../activations/tanh.h"
#include "../reductions/product.h"
#include "../reductions/sum.h"
#include "concat.h"
#include "identity.h"
#include "linear.h"
#include "select.h"
#include "sequential.h"
#include "reduce.h"

namespace nnlib
{

template <
	template <typename> class GateAct = Logistic,
	template <typename> class InAct = TanH,
	template <typename> class OutAct = TanH,
	typename T = double
>
class LSTM : public Module<T>
{
public:
	LSTM(size_t inps, size_t outs, size_t bats = 1) :
		m_nn(nullptr),
		m_inputBlame(bats, inps),
		m_output(bats, outs)
	{
		resize(inps, outs);
	}
	
	virtual void resize(size_t inps, size_t outs) override
	{
		Module<T>::resize(inps, outs);
		
		delete m_nn;
		
		// input = x(t) . y(t - 1) . h(t - 1)
		// output = y(t)
		m_nn = new Sequential<T>(
			new Concat<T>(
				// x(t) . y(t - 1) . h(t - 1)
				new Identity<T>(inps + 2 * outs),
				
				// inputGate(_) . forgetGate(_)
				new Sequential<T>(
					new Linear<T>(2 * outs),
					new Activation<GateAct, T>()
				),
				
				// inputActivation(x(t) . y(t - 1))
				new Sequential<T>(
					new Select<T>(0, inps + outs),
					new Linear<T>(outs),
					new Activation<InAct, T>()
				)
			),
			new Concat<T>(
				// x(t) . y(t - 1)
				new Select<T>(0, inps + outs),
				
				// h(t) = sum(product( h(t - 1) . inputGate(_) . forgetGate(_) . inputActivation(_) ))
				new Sequential<T>(
					new Select<T>(inps + outs, 4 * outs),
					new Reduce<Product, T>(2 * outs),
					new Reduce<Sum, T>(outs)
				)
			),
			new Concat<T>(
				// outputGate(_)
				new Sequential<T>(
					new Linear<T>(outs),
					new Activation<GateAct, T>()
				),
				
				// tanh(h(t))
				new Sequential<T>(
					new Select<T>(inps + outs, outs),
					new Activation<OutAct, T>()
				),
				
				// h(t)
				new Select<T>(inps + outs, outs)
			),
			new Concat<T>(
				// y(t)
				new Sequential<T>(
					new Select<T>(0, 2 * outs),
					new Reduce<Product, T>(outs)
				),
				
				// h(t)
				new Select<T>(2 * outs, outs)
			)
		);
	}
	
	/// Forward propagation of a sequence, resetting hidden state.
	/// \todo allow sequence batches
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		
		size_t sequenceLength = inputs.rows();
		size_t outs = m_output.cols();
		
		// reset hidden state and output
		m_nn->output().fill(0);
		
		// loop over blocks and forward propagate
		for(size_t i = 0; i < sequenceLength; ++i)
		{
			m_output(i).copy(Vector<T>(m_nn->forward(Vector<T>::concatenate(inputs(i), m_nn->output()))(0), 0, outs));
		}
		
		return m_output;
	}
	
	/// Backpropagation across a sequence.
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		/*
		size_t sequenceLength = inputs.rows();
		size_t blockSize = 1; // inputs.rows() / sequenceLength;
		
		// create an appropriately-sized view of inputs
		// Matrix<T> inp = inputs.block(0, 0, blockSize);
		/// \todo this is not the cleanest, fastest, or most stable way to do this... make it better
		Matrix<T> inp(blockSize, inputs.cols());
		inp(0).copy(inputs(0));
		
		Vector<T> foo, bar;
		
		for(size_t i = sequenceLength; i > 1; --i)
		{
			// send it forward again to reset internal state
			inp(0).copy(inputs(i - 1));
			foo.concatenate({ &inp, &m_outputs(i - 2), &m_hiddens(i - 2) });
			m_nn->forward(foo);
			
			// now we can backprop it
			m_nn->backward(foo, bar);
		}
		*/
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_output;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
private:
	Sequential<T> *m_nn;
	Matrix<T> m_inputBlame;
	Matrix<T> m_output;
};

}

#endif
