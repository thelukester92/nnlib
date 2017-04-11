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
		m_outputAndHidden(bats, 2 * outs),
		m_inputBlame(bats, inps),
		m_output(0, 0)
	{
		resize(inps, outs);
	}
	
	virtual void resize(size_t inps, size_t outs) override
	{
		m_outputAndHidden.resize(m_outputAndHidden.rows(), 2 * outs);
		m_inputBlame.resize(m_inputBlame.rows(), inps);
		m_output.resize(m_output.rows(), outs);
		m_outputAndHidden.block(m_output, 0, 0, (size_t) -1, outs);
		
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
	
	virtual void batch(size_t bats) override
	{
		m_outputAndHidden.resize(bats, m_outputAndHidden.cols());
		m_inputBlame.resize(bats, m_inputBlame.cols());
		m_output.resize(bats, m_output.cols());
		m_outputAndHidden.block(m_output, 0, 0, (size_t) -1, m_output.cols());
	}
	
	/// Forward propagation of a sequence, resetting hidden state.
	/// \todo allow sequence batches
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		
		size_t sequenceLength = inputs.rows();
		
		// reset hidden state and output
		m_nn->output().fill(0);
		
		// loop over blocks and forward propagate
		for(size_t i = 0; i < sequenceLength; ++i)
		{
			Vector<T> inp = Vector<T>::concatenate(inputs(i), m_nn->output());
			m_outputAndHidden(i).copy(m_nn->forward(inp)(0));
		}
		
		return m_output;
	}
	
	/// Backpropagation across a sequence.
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		NNAssert(blame.rows() == this->batchSize() && blame.cols() == this->outputs(), "Incompatible blame!");
		
		size_t sequenceLength = inputs.rows();
		size_t inps = inputs.cols(), outs = m_output.cols();
		
		// reset hidden blame
		Vector<T> blam(2 * outs, 0);
		Vector<T> inp(inps + 2 * outs);
		
		// loop over blocks and back propagate
		for(int t = sequenceLength - 1; t >= 0; --t)
		{
			// reset state to time t
			inp.narrow(inps).copy(inputs(t));
			if(t > 0)
				inp.narrow(outs * 2, inps).copy(m_outputAndHidden(t - 1));
			else
				inp.narrow(outs * 2).fill(0.0);
			
			m_nn->forward(inp);
			
			// backprop in this old state
			blam.narrow(outs).addScaled(blame(t));
			m_nn->backward(inp, blam);
			
			// update input blame
			m_inputBlame(t).copy(Vector<T>(m_nn->output()(0), 0, inps));
		}
		
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
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		return m_nn->parameters();
	}

	virtual Vector<Tensor<T> *> blame() override
	{
		return m_nn->blame();
	}
	
private:
	Sequential<T> *m_nn;
	Matrix<T> m_outputAndHidden;
	
	Matrix<T> m_inputBlame;
	Matrix<T> m_output;
};

}

#endif
