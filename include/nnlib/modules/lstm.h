#ifndef LSTM_H
#define LSTM_H

#include "../module.h"
#include "linear.h"
#include "tanh.h"
#include "logistic.h"
#include "random.h"

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
		m_giantNetwork = new Sequential<T>(
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
					new Split<T>(0, inputSize + outputSize),
					new Linear<T>(inputSize + outputSize, hiddenSize),
					new TanH<T>()
				)
			),
			new Concat<T>(
				// x(t) . y(t - 1)
				new Split<T>(0, inputSize + outputSize),
				
				// h(t)
				new Sequential<T>(
					new Split<T>(inputSize + outputSize, 4 * hiddenSize),
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
					new Split<T>(inputSize + outputSize, hiddenSize),
					new TanH<T>()
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
	Linear<T> m_inputGate, m_forgetGate, m_outputGate;
	Logistic<T> m_logistic;
	Tanh<T> m_tanh;
	
	Matrix<T> m_inputActs, m_forgetActs, m_outputActs, m_hiddens;
	Vector<T> m_temp;
	
	size_t m_step;
	
	
	
	
	
	
	
	LSTM(size_t inps, size_t outs, size_t batch = 1)
	: m_addBuffer(batch, 1),
	  m_bias(outs), m_weights(outs, inps),
	  m_biasBlame(outs), m_weightsBlame(outs, inps),
	  m_inputBlame(batch, inps), m_outputs(batch, outs)
	{
		resetWeights();
	}

	Linear(size_t outs)
	: m_addBuffer(1, 1),
	  m_bias(outs), m_weights(outs, 0),
	  m_biasBlame(outs), m_weightsBlame(outs, 0),
	  m_inputBlame(1, 0), m_outputs(1, outs)
	{
		resetWeights();
	}

	Vector<T> &bias()
	{
		return m_bias;
	}

	Matrix<T> &weights()
	{
		return m_weights;
	}

	void resetWeights()
	{
		for(auto &val : m_weights)
			val = Random<T>::normal(0, 1, 1);
		for(auto &val : m_bias)
			val = Random<T>::normal(0, 1, 1);
	}

	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		m_addBuffer.resize(bats).fill(1);
		m_inputBlame.resize(bats, inps);
		m_outputs.resize(bats, outs);
		m_bias.resize(outs);
		m_weights.resize(outs, inps);
		m_biasBlame.resize(outs);
		m_weightsBlame.resize(outs, inps);
		resetWeights();
	}

	virtual void batch(size_t size) override
	{
		m_addBuffer.resize(size).fill(1);
		m_inputBlame.resize(size, m_inputBlame.cols());
		m_outputs.resize(size, m_outputs.cols());
	}

	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		Matrix<T>::multiply(inputs, m_weights, m_outputs, false, true);
		Matrix<T>::addOuterProduct(m_addBuffer, m_bias, m_outputs);
		return m_outputs;
	}

	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		Matrix<T>::multiply(blame, inputs, m_weightsBlame, true, false, 1, 1);
		Matrix<T>::multiply(blame, m_addBuffer, m_biasBlame, true, 1, 1);
		Matrix<T>::multiply(blame, m_weights, m_inputBlame);
		return m_inputBlame;
	}

	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}

	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}

	virtual Vector<Tensor<T> *> parameters() override
	{
		return { &m_bias, &m_weights };
	}

	virtual Vector<Tensor<T> *> blame() override
	{
		return { &m_biasBlame, &m_weightsBlame };
	}

private:
	Vector<T> m_addBuffer;		///< A vector of 1s to quickly evaluate bias.
	Vector<T> m_bias;			///< The bias; adding constants to outputs.
	Matrix<T> m_weights;		///< The parameters.
	Vector<T> m_biasBlame;		///< Gradient of the error w.r.t. the bias.
	Matrix<T> m_weightsBlame;	///< Gradient of the error w.r.t. the weights.
	Matrix<T> m_inputBlame;		///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;		///< The output of this layer.
};

}

#endif
