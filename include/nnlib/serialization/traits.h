#ifndef TRAITS_H
#define TRAITS_H

#include <typeindex>
#include <type_traits>
#include <unordered_map>

// ignore -Wunused-value for this section; it is unused on purpose for SFINAE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

namespace nnlib
{

class SerializedNode;

namespace traits
{
	/// A struct containing the name of a class given the class type.
	template <typename>
	struct NameOf
	{
		static const std::string value;
	};
	
	/// A struct containing the base type of the given class.
	template <typename T>
	struct BaseOf
	{
		using type = T;
	};
	
	/// Check whether the given type has the load and save methods. Default is false.
	template <typename, typename = int, typename = int>
	struct HasLoadAndSave : std::false_type
	{};

	/// Check whether the given type has the load and save methods. This override determines it does.
	template <typename T>
	struct HasLoadAndSave<T, decltype(std::declval<T>().load(std::declval<const SerializedNode &>()), 0), decltype(std::declval<T>().save(std::declval<SerializedNode &>()), 0)> : std::true_type
	{};
}

}

// end ignore -Wunused-value
#pragma GCC diagnostic pop

#endif
