#ifndef SERIALIZATION_SERIALIZED_HPP
#define SERIALIZATION_SERIALIZED_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include "factory.hpp"
#include "traits.hpp"

namespace nnlib
{

/// \brief Serialized data.
///
/// Serialized is an intermediary value between classes and fully serialized data.
/// Serializers can load nodes from or save nodes to a file.
/// Classes can load from a node or save to a node.
/// Numbers are always stored as floating point values, but can be cast to any arithmetic type.
/// Arrays store sequences of node pointers and objects map strings to node pointers.
class Serialized
{
public:
	/// A tag indicating a node's data type.
	enum Type
	{
		Null, Boolean, Integer, Float, String, Array, Object
	};

	/// Create a node with the given type; default is Null.
	inline Serialized(Type t = Null);

	/// Create a node with the given value.
	template <typename ... Ts>
	inline Serialized(Ts && ...values);

	inline Serialized(const Serialized &other);
	inline Serialized(Serialized &other);
	inline Serialized(Serialized &&other);
	inline ~Serialized();

	inline Serialized &operator=(const Serialized &other);
	inline Serialized &operator=(Serialized &&other);

	inline Type type() const;
	inline void type(Type type);
	inline size_t size() const;

	// Primary Getters

	template <typename T>
	typename std::enable_if<std::is_integral<T>::value, T>::type get() const;

	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value, T>::type get() const;

	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, T>::type get() const;

	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type get() const;

	/// Self-getter for a consistent interface.
	///
	/// This method simply returns self. Without this overload,
	/// a null node self-getter is null rather than a null node.
	/// This method is not const because it can self-modify.
	template <typename T>
	typename std::enable_if<std::is_same<T, const Serialized *>::value, T>::type get() const;

	/// Self-getter for a consistent interface.
	///
	/// This method simply returns self. Without this overload,
	/// a null node self-getter is null rather than a null node.
	/// This method is not const because it can self-modify.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized *>::value, T>::type get();

	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && std::is_abstract<typename std::remove_pointer<T>::type>::value, T>::type get() const;

	template <typename T>
	typename std::enable_if<
		std::is_pointer<T>::value && !std::is_abstract<typename std::remove_pointer<T>::type>::value &&
		!std::is_same<typename std::remove_const<typename std::remove_pointer<T>::type>::type, Serialized>::value, T
	>::type get() const;

	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type get(T itr, const T &end) const;

	// Primary Setters

	template <typename T>
	typename std::enable_if<std::is_same<T, Type>::value>::type set(T type);

	template <typename T>
	typename std::enable_if<std::is_same<T, bool>::value>::type set(T value);

	template <typename T>
	typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type set(T value);

	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value>::type set(T value);

	template <typename T>
	typename std::enable_if<std::is_convertible<T, std::string>::value && !std::is_same<T, std::nullptr_t>::value>::type set(const T &value);

	template <typename T>
	typename std::enable_if<traits::HasSave<T>::value>::type set(const T &value);

	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && !std::is_convertible<T, std::string>::value>::type set(const T &value);

	template <typename T>
	typename std::enable_if<std::is_same<T, std::nullptr_t>::value>::type set(T value);

	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized>::value>::type set(const T &value);

	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type set(T itr, const T &end);

	// Array Operations / Convenience Methods

	template <typename T, typename ... Ts>
	void add(T && value, Ts && ...values);
	inline void add(Serialized *value);

	inline Type type(size_t i) const;
	inline void type(size_t i, Type type);
	inline size_t size(size_t i) const;

	template <typename T = const Serialized *>
	T get(size_t i) const;

	template <typename T = Serialized *>
	T get(size_t i);

	template <typename T>
	void get(size_t i, T itr, const T &end) const;

	template <typename T, typename ... Ts>
	void set(size_t i, T && first, Ts && ...values);

	// Object Operations / Convenience Methods

	inline bool has(const std::string &key) const;
	inline const std::vector<std::string> &keys() const;

	inline Type type(const std::string &key) const;
	inline void type(const std::string &key, Type type);
	inline size_t size(const std::string &key) const;

	template <typename T = const Serialized *>
	T get(const std::string &key) const;

	template <typename T = Serialized *>
	T get(const std::string &key);

	template <typename T>
	void get(const std::string &key, T itr, const T &end) const;

	template <typename T, typename ... Ts>
	void set(const std::string &key, T && first, Ts && ...values);

private:
	struct SerializedObject
	{
		std::unordered_map<std::string, Serialized *> map;
		std::vector<std::string> keys;
	};

	Type m_type;
	union
	{
		long long m_int;
		double m_float;
		std::string m_string;
		std::vector<Serialized *> m_array;
		SerializedObject m_object;
	};
};

}

#include "detail/serialized.tpp"

#endif
