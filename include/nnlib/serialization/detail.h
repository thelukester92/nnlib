#ifndef SERIALIZATION_DETAIL_H
#define SERIALIZATION_DETAIL_H

#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <functional>
#include "../error.h"

namespace nnlib
{

namespace detail
{

// ignore -Wunused-value for this section; it is unused on purpose for SFINAE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

/// Check whether the given type has the serialize method. Default is false.
template <typename T, typename = int>
struct HasSerialize : std::false_type
{};

/// Check whether the given type has the serialize method. This override determines it does.
template <typename T>
struct HasSerialize<T, decltype(&T::template serialize<T>, 0)> : std::true_type
{};

/// Check whether the given type has the load and save methods. Default is false.
template <typename T, typename = int, typename = int>
struct HasLoadAndSave : std::false_type
{};

/// Check whether the given type has the load and save methods. This override determines it does.
template <typename T>
struct HasLoadAndSave<T, decltype(&T::template load<T>, 0), decltype(&T::template save<T>, 0)> : std::true_type
{};

#pragma GCC diagnostic pop

/// \brief Hook in code before main.
///
/// Merely instantiating this template creates an object of type T.
/// This allows us to run code (the default contructor for T) before main.
/// This is useful for polymorphic serialization support.
template <typename T>
class Hook
{
public:
	static T &instance() { return m_instance; }
private:
	static T &create() { static T t; return t; }
	static T &m_instance;
};
template <typename T> T &Hook<T>::m_instance = Hook<T>::create();

/// A struct containing the registered base class.
template <typename T>
struct BaseOf
{
	using type = T;
};

/// A struct containing the class name.
template <typename T>
struct BindingName
{
	static const std::string value;
};

/// \brief Binding information for a polymorphic type.
///
/// This class contains constructors and serializers for polymorphic serialization.
template <typename Base>
class Binding
{
public:
	using Constructor = std::function<Base*()>;
	using CopyConstructor = std::function<Base*(const Base &)>;
	using Serializer = std::function<void(void*,Base*)>;
	
	template <typename Derived>
	static std::string bindDerived(const std::string &name)
	{
		instantiateBinding(static_cast<Derived *>(nullptr), 0);
		constructors().emplace(name, []() { return new Derived(); });
		bindingNames().emplace(typeid(Derived), name);
		Binding::bindCopyConstructor<Derived>(name);
		return name;
	}
	
	template <typename Archive, typename Derived>
	static void bindDerivedToArchive()
	{
		auto aKey = std::type_index(typeid(Archive));
		auto dKey = std::type_index(typeid(Derived));
		
		// insert or fetch the archive list
		auto &aMap = serializers()[aKey];
		
		// insert the derived serializer
		aMap.emplace(dKey, [](void *ar, Base *base)
		{
			reinterpret_cast<Archive *>(ar)->process(*dynamic_cast<Derived *>(base));
		});
	}
	
	static std::string bindingName(const std::type_info &info)
	{
		auto i = bindingNames().find(info);
		NNHardAssertNotEquals(i, bindingNames().end(), "Could not find binding for the given type!");
		return i->second;
	}
	
	static Base *construct(const std::string &name)
	{
		auto i = constructors().find(name);
		NNHardAssertNotEquals(i, constructors().end(), "Could not find constructor for " + name + "!");
		return i->second();
	}
	
	static Base *construct(const std::type_info &info)
	{
		return construct(bindingName(info));
	}
	
	static Base *constructLike(const Base *arg)
	{
		return construct(bindingName(typeid(*arg)));
	}
	
	static Base *constructCopy(const Base *arg)
	{
		std::string name = bindingName(typeid(*arg));
		auto i = copyConstructors().find(name);
		NNHardAssertNotEquals(i, copyConstructors().end(), "Could not find copy constructor for " + name + "!");
		return i->second(*arg);
	}
	
	template <typename Archive>
	static void serialize(Archive &ar, Base &base)
	{
		auto aKey = std::type_index(typeid(Archive));
		auto dKey = std::type_index(typeid(base));
		
		auto aItr = serializers().find(aKey);
		NNHardAssertNotEquals(aItr, serializers().end(), "Could not find binding for the given archive type!");
		
		auto dItr = aItr->second.find(dKey);
		NNHardAssertNotEquals(dItr, aItr->second.end(), "Could not find binding for the given base type!");
		
		dItr->second(&ar, &base);
	}
	
private:
	template <typename Derived>
	static typename std::enable_if<!std::is_copy_constructible<Derived>::value>::type bindCopyConstructor(const std::string &name)
	{}
	
	template <typename Derived>
	static typename std::enable_if<std::is_copy_constructible<Derived>::value>::type bindCopyConstructor(const std::string &name)
	{
		copyConstructors().emplace(name, [](const Base &arg)
		{
			return new Derived(static_cast<const Derived &>(arg));
		});
	}
	
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
	
	static std::unordered_map<std::type_index, std::string> &bindingNames()
	{
		static std::unordered_map<std::type_index, std::string> map;
		return map;
	}
	
	static std::unordered_map<std::type_index, std::unordered_map<std::type_index, Serializer>> &serializers()
	{
		static std::unordered_map<std::type_index, std::unordered_map<std::type_index, Serializer>> map;
		return map;
	}
};

/// Empty template to force instantiation of unused method templates.
template <void(*)()>
struct ForceInstantiation {};

/// The one place we know the true type of both Archive and T.
template <typename Archive, typename T>
struct PolymorphicSerializationSupport
{
	/// pass along this archive/type binding
	PolymorphicSerializationSupport()
	{
		Binding<typename BaseOf<T>::type>::template bindDerivedToArchive<Archive, T>();
	}
	
	/// never called, but its existance will run the constructor before main
	static void instantiate()
	{
		Hook<PolymorphicSerializationSupport>::instance();
	}
	
	typedef ForceInstantiation<instantiate> unused;
};

} // namespace detail

/// \brief The preferred overload for instantiateBinding.
///
/// Archives will create specializations of this function that will fail,
/// but the attempt to instantiate them will hook code in mefore main.
template <typename T>
void instantiateBinding(T*, int) {}

/// \brief Make a deep copy of the given base class.
///
/// This will use registered polymorphic types to clone the derived type through
/// a pointer to the base class.
template <typename Base>
Base *copy(const Base *arg)
{
	return detail::Binding<Base>::constructCopy(arg);
}

} // namespace nnlib

/// \brief Register a new archive type.
///
/// The substitution will fail (PolymorphicSerializationSupport has no type),
/// but the substitution will instantiate a static object, which will
/// enable Archive to serialize all registered types.
/// Must be placed outside namespace nnlib.
#define NNRegisterArchive(Archive) \
namespace nnlib { \
	template <typename T> \
	typename detail::PolymorphicSerializationSupport<Archive, T>::type instantiateBinding(T*, Archive*); \
}

/// \brief Register a new polymorphic type.
///
/// Must be placed outside namespace nnlib.
/// This is used both for serialization and deep copying.
#define NNRegisterType(Derived, Base) \
namespace nnlib { namespace detail { \
	template <> struct BaseOf<Derived> { using type = Base; }; \
	template <> const std::string BindingName<Derived>::value = Binding<Base>::bindDerived<Derived>(#Derived); \
} }

#endif
