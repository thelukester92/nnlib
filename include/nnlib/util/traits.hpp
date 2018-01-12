#ifndef UTIL_TRAITS_HPP
#define UTIL_TRAITS_HPP

#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include "../core/type.hpp"

// ignore -Wunused-value for this section; it is unused on purpose for SFINAE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

namespace nnlib
{

class Serialized;

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
	template <typename, typename = int>
	struct HasSave : std::false_type
	{};

	/// Check whether the given type has the load and save methods. This override determines it does.
	template <typename T>
	struct HasSave<T, decltype(std::declval<T>().save(std::declval<Serialized &>()), 0)> : std::true_type
	{};

	/// Check whether the given type has the load and save methods. Default is false.
	template <typename, typename = int, typename = int>
	struct HasLoadAndSave : std::false_type
	{};

	/// Check whether the given type has the load and save methods. This override determines it does.
	template <typename T>
	struct HasLoadAndSave<T, decltype(T(std::declval<const Serialized &>()), 0), decltype(std::declval<T>().save(std::declval<Serialized &>()), 0)> : std::true_type
	{};

	/// Check whether the given type has the begin and end methods. Default is false.
	template <typename, typename = int, typename = int>
	struct HasBeginAndEnd : std::false_type
	{};

	/// Check whether the given type has the begin and end methods. This override determines it does.
	template <typename T>
	struct HasBeginAndEnd<T, decltype(std::declval<T>().begin(), 0), decltype(std::declval<T>().end(), 0)> : std::true_type
	{};
}

}

// end ignore -Wunused-value
#pragma GCC diagnostic pop

#endif
