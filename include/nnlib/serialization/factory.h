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
	{
		static const std::string value;
	};
}

template <typename Base>
class Factory
{
public:
	using Constructor = std::function<Base*()>;
	
	template <typename Derived>
	static std::string registerDerivedType(const std::string &name)
	{
		return registerDerivedType<Derived>(name, []() { return new Derived(); });
	}
	
	template <typename Derived>
	static std::string registerDerivedType(const std::string name, const Constructor &ctor)
	{
		constructors().emplace(name, ctor);
		return name;
	}
	
	static Base *construct(const std::string &name)
	{
		return constructors()[name]();
	}
	
	template <typename Derived>
	static SerializedNode save(const Derived &value)
	{
		SerializedNode node;
		node.set("type", detail::NameOf<Derived>::value);
		node.set("value", value);
		return node;
	}
	
	/// \todo document this whole file and make the following function look nicer
	static Base *load(const SerializedNode &node)
	{
		Base *value = construct(node.get<std::string>("type"));
		value->load(node.get<SerializedNode>("value"));
		return value;
	}
	
private:
	static std::unordered_map<std::string, Constructor> &constructors()
	{
		static std::unordered_map<std::string, Constructor> map;
		return map;
	}
};

}

#define NNRegisterType(Derived, Base) \
namespace nnlib { namespace detail { \
	template <> const std::string NameOf<Derived>::value = Factory<Base>::template registerDerivedType<Derived>(#Derived); \
} }

#endif
