#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "module.h"

namespace nnlib
{

/// \todo allow a critic as the last "layer"

/// A feed-forward neural network.
/// Takes ownership of all modules added to it.
template <typename T>
class Sequential : public Module<T>
{
using Module<T>::m_inputBlame;
using Module<T>::m_output;
public:
	Sequential() : Module<T>(0, 0, 0), m_modules(0)
	{}
	
	~Sequential()
	{
		for(auto i : m_modules)
			delete i;
	}
	
	/// Add a module to this network.
	/// Network takes ownership of the new module.
	void add(Module<T> *module)
	{
		m_modules.push_back(module);
		m_inputBlame.share(m_modules.front()->inputBlame());
		m_output.share(m_modules.back()->output());
	}
	
	/// Add multiple modules at once.
	template <typename ... Ts>
	void add(Module<T> *module, Ts*... more)
	{
		add(module);
		add(more...);
	}
	
	/// Get the module at index i.
	Module<T> &module(size_t i)
	{
		return *m_modules[i];
	}
	
	/// Get the number of modules.
	size_t modules() const
	{
		return m_modules.size();
	}
	
	/// Release the module at index i from ownership.
	/// Caller becomes responsible for deleting the module.
	Module<T> *release(size_t i)
	{
		Module<T> *module = m_modules[i];
		m_modules.erase(i);
		if(m_modules.size() > 0)
		{
			m_inputBlame.share(m_modules.front()->inputBlame());
			m_output.share(m_modules.back()->output());
		}
		else
		{
			m_inputBlame.resize(0, 0);
			m_output.resize(0, 0);
		}
		return module;
	}
	
	/// Feed in input vectors and return cached output vectors.
	virtual Matrix<T> &forward(const Matrix<T> &input) override
	{
		NNAssert(m_modules.size() > 0, "Cannot forward propagate in an empty network!");
		const Matrix<T> *in = &input;
		for(auto i : m_modules)
			in = &i->forward(*in);
		return m_modules.back()->output();
	}
	
	/// Feed in inputs and output blames (gradient) and return cached input blame vectors.
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) override
	{
		NNAssert(m_modules.size() > 0, "Cannot backpropagate in an empty network!");
		const Matrix<T> *bl = &blame;
		for(size_t i = m_modules.size() - 1; i > 0; --i)
			bl = &m_modules[i]->backward(m_modules[i - 1]->output(), *bl);
		return m_modules[0]->backward(input, *bl);
	}
	
	/// Return pointers to all parameters (i.e. for flattening).
	virtual Vector<Tensor<T> *> parameters() override
	{
		Vector<Tensor<T> *> params;
		for(auto i : m_modules)
			for(auto j : i->parameters())
				params.push_back(j);
		return params;
	}
	
	/// Return pointers to parameter blame buffers (i.e. for flattening).
	virtual Vector<Tensor<T> *> blame() override
	{
		Vector<Tensor<T> *> blam;
		for(auto i : m_modules)
			for(auto j : i->blame())
				blam.push_back(j);
		return blam;
	}

private:
	Vector<Module<T> *> m_modules;
};

}

#endif
