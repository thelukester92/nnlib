#ifndef FACTORY_H
#define FACTORY_H

#include <functional>
#include <string>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include "serialized.h"
#include "traits.h"

namespace nnlib
{

/// \brief A factory class for creating instances of derived classes (D) through pointers to a base class (B).
///
/// This is primarily used for seriailzation, but is also used for deep copying of modules.
/// A convenience macro is provided, NNRegisterType(Derived, Base), so Factory does not usually need to be used directly.
template <typename B>
class Factory
{
public:
	using Constructor = std::function<B*()>;
	using CopyConstructor = std::function<B*(const B &)>;
	
	/// All methods are static; instances are not needed.
	Factory() = delete;
	
	/// Register a derived type using default constructor.
	template <typename D>
	static typename std::enable_if<std::is_default_constructible<D>::value, std::string>::type
	registerDerivedType(const std::string &name)
	{
		return registerDerivedType<D>(name, []() { return new D(); });
	}
	
	/// Register a derived type given a specific constructor; copy constructible.
	template <typename D>
	static typename std::enable_if<std::is_copy_constructible<D>::value, std::string>::type
	registerDerivedType(const std::string &name, const Constructor &ctor)
	{
		constructors().emplace(name, ctor);
		copyConstructors().emplace(name, [](const B &b) { return new D(static_cast<const D &>(b)); });
		derivedNames().emplace(typeid(D), name);
		return name;
	}
	
	/// Register a derived type given a specific constructor; not copy constructible.
	template <typename D>
	static typename std::enable_if<!std::is_copy_constructible<D>::value, std::string>::type
	registerDerivedType(const std::string &name, const Constructor &ctor)
	{
		constructors().emplace(name, ctor);
		derivedNames().emplace(typeid(D), name);
		return name;
	}
	
	/// Construct an instance of a derived class by registered class name.
	static B *construct(const std::string &name)
	{
		auto i = constructors().find(name);
		NNHardAssertNotEquals(i, constructors().end(), "Attempted to construct an unregistered type!");
		return i->second();
	}
	
	/// Construct a copy of a derived class instance through a base class pointer.
	static B *constructCopy(const B *original)
	{
		auto i = derivedNames().find(typeid(*original));
		NNHardAssertNotEquals(i, derivedNames().end(), "Attempted to copy-construct an unregistered type!");
		
		auto j = copyConstructors().find(i->second);
		NNHardAssertNotEquals(j, copyConstructors().end(), "Attempted to copy-construct an unregistered type!");
		
		return j->second(*original);
	}
	
	static std::string derivedName(const std::type_index &idx)
	{
		auto i = derivedNames().find(idx);
		NNHardAssertNotEquals(i, derivedNames().end(), "Attempted to get derived name of an unregistered type!");
		return i->second;
	}
	
private:
	/// Static map of constructors.
	static std::unordered_map<std::string, Constructor> &constructors()
	{
		static std::unordered_map<std::string, Constructor> map;
		return map;
	}
	
	/// Static map of copy constructors.
	static std::unordered_map<std::string, CopyConstructor> &copyConstructors()
	{
		static std::unordered_map<std::string, CopyConstructor> map;
		return map;
	}
	
	/// Static map of derived names from type indices.
	static std::unordered_map<std::type_index, std::string> &derivedNames()
	{
		static std::unordered_map<std::type_index, std::string> map;
		return map;
	}
};

}

/// \brief Register a derived type for serialization and copy-construction.
///
/// This should only be used once for each Derived class; serialization through multiple
/// base classes is not supported.
/// The call to this macro must be placed outside namespace nnlib.
#define NNRegisterType(Derived, Base) \
namespace nnlib { namespace traits { \
	template <> const std::string NameOf<Derived>::value = Factory<Base>::template registerDerivedType<Derived>(#Derived); \
	template <> struct BaseOf<Derived> { using type = Base; }; \
} }

#endif
