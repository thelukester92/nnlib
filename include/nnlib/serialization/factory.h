#ifndef FACTORY_H
#define FACTORY_H

#include <functional>
#include <string>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include "serialized_node.h"
#include "traits.h"

namespace nnlib
{

template <typename B>
class Factory
{
public:
	using Constructor = std::function<B*()>;
	using CopyConstructor = std::function<B*(const B &)>;
	
	template <typename D>
	static typename std::enable_if<std::is_default_constructible<D>::value && std::is_copy_constructible<D>::value, std::string>::type
	registerDerivedType(const std::string &name)
	{
		return registerDerivedType<D>(name, []() { return new D(); }, [](const B &b) { return new D(static_cast<const D &>(b)); });
	}
	
	template <typename D>
	static typename std::enable_if<std::is_default_constructible<D>::value && !std::is_copy_constructible<D>::value, std::string>::type
	registerDerivedType(const std::string &name)
	{
		return registerDerivedType<D>(name, []() { return new D(); });
	}
	
	template <typename D>
	static std::string registerDerivedType(const std::string &name, const Constructor &ctor)
	{
		constructors().emplace(name, ctor);
		derivedNames().emplace(typeid(D), name);
		return name;
	}
	
	template <typename D>
	static std::string registerDerivedType(const std::string &name, const Constructor &ctor, const CopyConstructor &copyCtor)
	{
		constructors().emplace(name, ctor);
		copyConstructors().emplace(name, copyCtor);
		derivedNames().emplace(typeid(D), name);
		return name;
	}
	
	static B *construct(const std::string &name)
	{
		return constructors()[name]();
	}
	
	static B *constructCopy(const B *original)
	{
		return copyConstructors()[derivedNames()[typeid(*original)]](*original);
	}
	
private:
	static std::unordered_map<std::string, Constructor> &constructors()
	{
		static std::unordered_map<std::string, Constructor> map;
		return map;
	}
	
	static std::unordered_map<std::string, CopyConstructor> &copyConstructors()
	{
		static std::unordered_map<std::string, CopyConstructor> map;
		return map;
	}
	
	static std::unordered_map<std::type_index, std::string> &derivedNames()
	{
		static std::unordered_map<std::type_index, std::string> map;
		return map;
	}
};

}

#define NNRegisterType(Derived, Base) \
namespace nnlib { namespace traits { \
	template <> const std::string NameOf<Derived>::value = Factory<Base>::template registerDerivedType<Derived>(#Derived); \
} }

#endif
