#ifndef NN_CONTAINER_HPP
#define NN_CONTAINER_HPP

#include "module.hpp"

namespace nnlib
{

/// The abtract base class for neural network modules that are made up of sub-modules.
template <typename T = double>
class Container : public Module<T>
{
public:
	template <typename ... Ms>
	Container(Ms... components) :
		m_components({ static_cast<Module<T> *>(components)... })
	{}
	
	Container(const Container &module);
	Container(const Serialized &node);
	
	Container &operator=(const Container &module);
	
	virtual ~Container();
	virtual void training(bool training = true) override;
	virtual void forget() override;
	virtual void save(Serialized &node) const override;
	
	/// Get a specific component from this container.
	Module<T> *component(size_t index);
	
	/// Get the number of components in this container.
	size_t components() const;
	
	/// Add multiple components to this container.
	template <typename ... Ms>
	Container &add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	/// Add a component to this container.
	virtual Container &add(Module<T> *component);
	
	/// Remove and return a specific component from this container. Caller is responsible for deleting this module.
	virtual Module<T> *remove(size_t index);
	
	/// Remove all components from this container and delete them.
	virtual Container &clear();
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor<T> *> paramsList() override;
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override;
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor<T> *> stateList() override;
	
protected:
	Storage<Module<T> *> m_components;
};

}

NNRegisterType(Container, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Container<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/container.tpp"
#endif

#endif
