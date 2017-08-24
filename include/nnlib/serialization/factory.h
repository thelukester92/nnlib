#ifndef FACTORY_H
#define FACTORY_H

namespace nnlib
{

#include <functional>
#include <string>
#include <unordered_map>
#include "serialized_node.h"

namespace detail
{
	template <typename T>
	struct NameOf
	{};
}

template <typename Base>
class Factory
{
public:
	using Constructor = std::function<Base*()>;
	
	template <typename Derived>
	static void registerDerivedType(const std::string &name)
	{
		registerDerivedType<Derived>(name, []() { return new Derived(); });
	}
	
	template <typename Derived>
	static void registerDerivedType(const std::string name, const Constructor &ctor)
	{
		constructors().emplace(name, ctor);
	}
	
	static Base *construct(const std::string &name)
	{
		return constructors()[name]();
	}
	
	template <typename Derived>
	static SerializedNode save(const Derived &value)
	{
		SerializedNode node;
		node.set("type", detail::NameOf<Derived>::value());
		node.set("value", value);
		return node;
	}
	
	/// \todo document this whole file and make the following function look nicer
	static Base *load(const SerializedNode &node)
	{
		Base *value = construct(node.get<std::string>("type"));
		value->load(*node.as<SerializedNode::Object>().at("value"));
		return value;
	}
	
private:
	static std::unordered_map<std::string, Constructor> &constructors()
	{
		static std::unordered_map<std::string, Constructor> map;
		return map;
	}
};

namespace detail
{
	template <typename Derived, typename Base>
	struct Registerer
	{
		Registerer(const std::string &name)
		{
			Factory<Base>::template registerDerivedType<Derived>(name);
		}
	};
}

}

#define NNConcatHelper(a, b) a##b
#define NNConcat(a, b) NNConcatHelper(a, b)

#define NNRegisterType(Derived, Base) \
namespace nnlib { namespace detail { \
	static Registerer<Derived, Base> NNConcat(registerer, __COUNTER__)(#Derived); \
	template <> struct NameOf<Derived> { static std::string value() { return #Derived; } }; \
} }

#endif
