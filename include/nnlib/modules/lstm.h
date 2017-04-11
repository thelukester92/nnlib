#ifndef LSTM_H
#define LSTM_H

#include "../module.h"
#include "../util/random.h"
#include "../activations/logistic.h"
#include "../activations/tanh.h"
#include "concat.h"
#include "identity.h"
#include "linear.h"
#include "select.h"
#include "sequential.h"

namespace nnlib
{

template <typename T = double>
class LSTM : public Module<T>
{
public:
	void setup()
	{
		/// \todo implement Split, a module which takes an offset and a size and filters out all other inputs
		/// \todo save a pointer to the Sequential that produces h(t) for accessing h(t) later
		/// \todo determine how this is supposed to work when doing BPTT...?
		
		// input = x(t) . y(t - 1) . h(t - 1)
		// output = y(t)
		auto *m_giantNetwork = new Sequential<T>(
			new Concat<T>(
				// x(t) . y(t - 1) . h(t - 1)
				new Identity<T>(),
				
				// forgetGate(_) . inputGate(_)
				new Sequential<T>(
					new Linear<T>(inputSize + outputSize + hiddenSize, 2 * hiddenSize),
					new Logistic<T>()
				),
				
				// inputActivation(x(t) . y(t - 1))
				new Sequential<T>(
					new Select<T>(0, inputSize + outputSize),
					new Linear<T>(inputSize + outputSize, hiddenSize),
					new Activation<TanH, T>()
				)
			),
			new Concat<T>(
				// x(t) . y(t - 1)
				new Select<T>(0, inputSize + outputSize),
				
				// h(t)
				new Sequential<T>(
					new Select<T>(inputSize + outputSize, 4 * hiddenSize),
					new ProductPool<T>(),
					new SumPool<T>()
				)
			),
			new Concat<T>(
				// outputGate(_)
				new Sequential<T>(
					new Linear<T>(inputSize + outputSize + hiddenSize, hiddenSize),
					new Logistic<T>()
				),
				
				// tanh(h(t))
				new Sequential<T>(
					new Select<T>(inputSize + outputSize, hiddenSize),
					new Activation<TanH, T>()
				)
			),
			// y(t)
			new ProductPool<T>()
		);
	}
	
	void reset()
	{
		m_hiddens[0].fill(0.0);
		m_outputs[0].fill(0.0);
		m_step = 0;
	}
	
	void stepForward(const Vector<T> &x, const Vector<T> &yPrev, const Vector<T> &hPrev, Vector<T> &y, Vector<T> &h)
	{
		size_t hids = x.size();
		
		Vector<T> xyh = Vector<T>::concatenate(x, yPrev, hPrev);
		Vector<T> xy = xyh.narrow(2 * hids);
		
		h.copy(
			m_adder.forward(
				m_multiplier.forward(
					m_inputGate.forward(xyh),
					m_inputActivation.forward(xy)
				),
				m_multiplier.forward(
					m_forgetGate.forward(xyh),
					hPrev
				)
			);
		);
		
		xyh.narrow(hids, 2 * hids).copy(h);
		
		y.copy(
			m_multiplier.forward(
				m_outputGate.forward(xyh),
				m_outputActivation.forward(h)
			)
		);
	}
	
	void stepBackward()
	{
		size_t hids = x.size();
		
		Vector<T> xyh = Vector<T>::concatenate(x, yPrev, hPrev);
		Vector<T> xy = xyh.narrow(2 * hids);
		
		// we want:
		// - update blame in the gates
		// - update hidden state blame
		// - update input blame
		
		oBlame = m_multiplier.forward(
			blame,
			m_outputActivation.forward(h)
		);
		
		h.copy(
			m_adder.forward(
				m_multiplier.forward(
					m_inputGate.forward(xyh),
					m_inputActivation.forward(xy)
				),
				m_multiplier.forward(
					m_forgetGate.forward(xyh),
					hPrev
				)
			);
		);
		
		xyh.narrow(hids, 2 * hids).copy(h);
		
		y.copy(
			m_multiplier.forward(
				m_outputGate.forward(xyh),
				m_outputActivation.forward(h)
			)
		);
	}
	
	
	
	/// Forward propagation of a sequence, resetting hidden state.
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		reset();
		
		for(size_t i = 0; i < inputs.rows(); ++i)
		{
			stepForward(inputs[i]);
		}
		
		return m_outputs;
	}
	
	/// Backpropagation across a sequence.
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		// reset old blame
		/// \todo ???
		
		for(m_step = inputs.rows(); m_step != 0; --m_step)
		{
			tmp.copy(m_outputs[m_step - 1]);
			activatedHidden;
			
			
		}
	}
	
private:
	
};

}

#endif
