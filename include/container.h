#ifndef CONTAINER_H
#define CONTAINER_H

#include "module.h"

namespace nnlib
{

/// Base class for modules that are made up of multiple modules.
/// Takes ownership of all modules added to it.
template <typename T>
class Container : public Module<T>
{
public:
	Container() : Module<T>(0, 0), m_modules(0)
	{}
	
	virtual ~Container()
	{
		for(auto i : m_modules)
			delete i;
	}
	
	virtual size_t inputCount() const override
	{
		return m_modules[0]->inputCount();
	}
	
	virtual size_t outputCount() const override
	{
		return m_modules.back()->outputCount();
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_modules[0]->inputBlame();
	}
	
	virtual const Matrix<T> &inputBlame() const override
	{
		return m_modules[0]->inputBlame();
	}
	
	virtual Matrix<T> &output() override
	{
		return m_modules.back()->output();
	}
	
	virtual const Matrix<T> &output() const override
	{
		return m_modules.back()->output();
	}
	
	/// Resize this module.
	virtual void resize(size_t inps, size_t outs)
	{
		m_inputBlame.resize(0, inps);
		m_output.resize(0, outs);
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
		return module;
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

protected:
	Vector<Module<T> *> m_modules;
};

}

#endif
