#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "module.h"

namespace nnlib
{

/// A feed-forward neural network.
/// Takes ownership of all modules added to it.
template <typename T>
class Sequential : public Module<T>
{
public:
	Sequential() : m_modules(0)
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
	}
	
	/// Add multiple modules at once.
	template <typename ... Ts>
	void add(Module<T> *module, Ts*... more)
	{
		add(module);
		add(more...);
	}
	
	/// Release the module at index i from ownership.
	/// Caller becomes responsible for deleting the module.
	Module<T> *release(size_t i)
	{
		Module<T> *module = m_modules[i];
		m_modules.erase(i);
		return module;
	}
	
	/// Feed in an input vector and return a cached output vector.
	virtual Vector<T> &forward(const Vector<T> &input) override
	{
		Assert(m_modules.size() > 0, "Cannot forward propagate in an empty network!");
		const Vector<T> *in = &input;
		for(auto i : m_modules)
			in = &i->forward(*in);
		return m_modules.back()->output();
	}
	
	/// Feed in an input and output blame (gradient) and return a cached input blame vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) override
	{
		Assert(m_modules.size() > 0, "Cannot backpropagate in an empty network!");
		const Vector<T> *bl = &blame;
		for(size_t i = m_modules.size() - 1; i > 0; --i)
			bl = &m_modules[i]->backward(m_modules[i - 1]->output(), *bl);
		return m_modules[0]->backward(input, *bl);
	}
	
	/// Get the input blame (gradient) buffer.
	virtual Vector<T> &inputBlame() override
	{
		return m_modules[0]->inputBlame();
	}
	
	/// Get the output buffer.
	virtual Vector<T> &output() override
	{
		return m_modules.back()->output();
	}

private:
	Vector<Module<T> *> m_modules;
};

}

#endif
