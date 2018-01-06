#ifndef SERIALIZATION_SERIALIZED_HPP
#define SERIALIZATION_SERIALIZED_HPP

#include <string>
#include <tuple>
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
	using SerializedArray = std::vector<Serialized>;
	using SerializedObject = std::pair<std::unordered_map<std::string, Serialized>, std::vector<std::string>>;

	/// A tag indicating a node's data type.
	enum Type
	{
		Null, Boolean, Integer, Float, String, Array, Object
	};

	/// Create a node with the given type; default is Null.
	explicit Serialized(Type t = Null) : m_type(Null)
	{
		type(t);
	}

	/// Create a node with the given value.
	template <typename ... Ts>
	explicit Serialized(Ts && ...values) :
		m_type(Null)
	{
		set(std::forward<Ts>(values)...);
	}

	/// Copy constructor.
	Serialized(const Serialized &other) :
		m_type(Null)
	{
		*this = other;
	}

	/// Non-const copy constructor (needed to avoid template conflict).
	Serialized(Serialized &other) :
		m_type(Null)
	{
		*this = other;
	}

	/// Move constructor.
	Serialized(Serialized &&other) :
		m_type(other.m_type)
	{
		switch(m_type)
		{
		case Null:
			break;
		case Boolean:
			m_bool = other.m_bool;
			break;
		case Integer:
			m_int = other.m_int;
			break;
		case Float:
			m_float = other.m_float;
			break;
		case String:
			m_string = std::move(other.m_string);
			break;
		case Array:
			m_array = std::move(other.m_array);
			break;
		case Object:
			m_object = std::move(other.m_object);
			break;
		}
	}

	/// Destructor.
	~Serialized()
	{
		// changing the type to null will call the appropriate union destructor
		type(Null);
	}

	/// Assignment operator.
	Serialized &operator=(const Serialized &other)
	{
		if(this != &other)
		{
			type(other.m_type);
			switch(m_type)
			{
			case Null:
				break;
			case Boolean:
				m_bool = other.m_bool;
				break;
			case Integer:
				m_int = other.m_int;
				break;
			case Float:
				m_float = other.m_float;
				break;
			case String:
				m_string = other.m_string;
				break;
			case Array:
				m_array = other.m_array;
				break;
			case Object:
				m_object = other.m_object;
				break;
			}
		}
		return *this;
	}

	/// Get the type of value stored.
	Type type() const
	{
		return m_type;
	}

	/// \brief Set the type of value stored.
	///
	/// Because some of the unioned types are classes, this explicitly
	/// calls the constructor and destructor when switching between them.
	/// See http://en.cppreference.com/w/cpp/language/union.
	void type(Type type)
	{
		if(type == m_type)
			return;

		if(m_type == String)
			m_string.~basic_string<char>();
		else if(m_type == Array)
			m_array.~SerializedArray();
		else if(m_type == Object)
			m_object.~SerializedObject();

		if(type == Type::String)
			new (&m_string) std::string;
		else if(type == Type::Array)
			new (&m_array) SerializedArray;
		else if(type == Type::Object)
			new (&m_object) SerializedObject;

		m_type = type;
	}

	/// \brief Get the size of the value stored.
	///
	/// For object and array, this is the number of elements contained.
	/// For other types, this returns 1.
	size_t size() const
	{
		if(m_type == Array)
			return m_array.size();
		else if(m_type == Object)
			return m_object.size();
		else
			return 1;
	}

// MARK: Getters

	/// Get a boolean value.
	template <typename T>
	typename std::enable_if<std::is_same<T, bool>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Boolean, "Invalid type!");
		return m_bool;
	}

	/// Get an integer value.
	template <typename T>
	typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Integer, "Invalid type!");
		return m_int;
	}

	/// Get a floating point value.
	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value, T>::type as() const
	{
		NNHardAssert(m_type == Float || m_type == Integer, "Invalid type!");
		if(m_type == Float)
			return m_float;
		else
			return m_int;
	}

	/// Get a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, String, "Invalid type!");
		return m_string;
	}

	/// Get a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, T &>::type as()
	{
		NNHardAssertEquals(m_type, String, "Invalid type!");
		return m_string;
	}

	/// Get an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedArray>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array;
	}

	/// Get an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedArray>::value, T &>::type as()
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array;
	}

	/// Get an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedObject>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object;
	}

	/// Get an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedObject>::value, T &>::type as()
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object;
	}

	/// Get a serializable value.
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type as() const
	{
		if(m_type == Object && m_object.has("polymorphic"))
			return T(*get<Serialized *>("data"));
		else
			return T(*this);
	}

	/// Get a serializable value (as a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && std::is_abstract<typename std::remove_pointer<T>::type>::value, T>::type as() const
	{
		if(m_type == Null)
			return nullptr;
		NNHardAssert(m_type == Object && m_object.has("polymorphic"), "Expected a polymorphic type!");
		return static_cast<T>(Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(get<std::string>("type"), *get<Serialized *>("data")));
	}

	/// Get a serializable value (as a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && !std::is_abstract<typename std::remove_pointer<T>::type>::value, T>::type as() const
	{
		if(m_type == Null)
			return nullptr;
		else if(m_type == Object && m_object.has("polymorphic"))
			return static_cast<T>(Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(get<std::string>("type"), *get<Serialized *>("data")));
		else
			return new typename std::remove_pointer<T>::type (*this);
	}

	/// Convert to a boolean value.
	template <typename T>
	typename std::enable_if<std::is_same<T, bool>::value, T>::type convertTo() const
	{
		switch(m_type)
		{
		case Null:
			return false;
		case Boolean:
			return m_bool;
		case Integer:
			return m_int;
		case Float:
			return m_float;
		default:
			throw Error("Invalid type!");
		}
	}

	/// Convert to an integer value.
	template <typename T>
	typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, T>::type convertTo() const
	{
		switch(m_type)
		{
		case Null:
			return 0;
		case Boolean:
			return m_bool;
		case Integer:
			return m_int;
		case Float:
			return m_float;
		default:
			throw Error("Invalid type!");
		}
	}

	/// Convert to a floating point value.
	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value, T>::type convertTo() const
	{
		switch(m_type)
		{
		case Null:
			return 0;
		case Boolean:
			return m_bool;
		case Integer:
			return m_int;
		case Float:
			return m_float;
		default:
			throw Error("Invalid type!");
		}
	}

	/// Convert to a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, T>::type convertTo() const
	{
		switch(m_type)
		{
		case Null:
			return "null";
		case Boolean:
			return m_bool ? "true" : "false";
		case Integer:
			return std::to_string(m_int);
		case Float:
			return std::to_string(m_float);
		case String:
			return m_string;
		default:
			throw Error("Invalid type!");
		}
	}

// MARK: Setters

	/// Set a boolean value.
	template <typename T>
	typename std::enable_if<std::is_same<T, bool>::value>::type set(T value)
	{
		type(Boolean);
		m_bool = value;
	}

	/// Set an integer value.
	template <typename T>
	typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type set(T value)
	{
		type(Integer);
		m_int = value;
	}

	/// Set a floating point value.
	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value>::type set(T value)
	{
		type(Float);
		m_float = value;
	}

	/// Set a string value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, std::string>::value && !std::is_same<T, std::nullptr_t>::value>::type set(const T &value)
	{
		type(String);
		m_string = value;
	}

	/// Set an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedArray>::value>::type set(const T &value)
	{
		type(Array);
		m_array = value;
	}

	/// Set an array value from a pair of iterators.
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type set(T i, const T &end)
	{
		type(Array);
		m_array.clear();
		while(i != end)
			m_array.add(new Serialized(*(i++)));
	}

	/// Set an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedObject>::value>::type set(const T &value)
	{
		type(Object);
		m_object = value;
	}

	/// Set a serializable value (from a reference).
	template <typename T>
	typename std::enable_if<traits::HasSave<T>::value>::type set(const T &value)
	{
		if(Factory<typename traits::BaseOf<T>::type>::isRegistered(typeid(value)))
		{
			set("polymorphic", true);
			set("type", Factory<typename traits::BaseOf<T>::type>::derivedName(typeid(value)));
			set("data", new Serialized());
			value.save(*m_object["data"]);
		}
		else
			value.save(*this);
	}

	/// Set a serializable value (from a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && !std::is_convertible<T, std::string>::value>::type set(const T value)
	{
		if(value == nullptr)
			type(Null);
		else
			set(*value);
	}

	/// Set a null pointer.
	void set(std::nullptr_t value)
	{
		type(Null);
	}

	/// Assignment.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized>::value>::type set(const T &value)
	{
		*this = value;
	}

// MARK: Array methods

	/// Get the type of an element in an array.
	Type type(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->type();
	}

	/// Get a numeric value from an array.
	template <typename T>
	typename std::enable_if<(std::is_arithmetic<T>::value || traits::HasLoadAndSave<T>::value) && !std::is_pointer<T>::value, T>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}

	/// Get a non-numeric value from an array.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !std::is_pointer<T>::value && !traits::HasLoadAndSave<T>::value, const T &>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}

	/// Get a non-numeric value from an array.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !std::is_pointer<T>::value && !traits::HasLoadAndSave<T>::value, T &>::type get(size_t i)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}

	/// Get a pointer value from an array.
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && !std::is_same<T, Serialized *>::value, T>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}

	/// Get a node from an array.
	template <typename T = Serialized *>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const Serialized *>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i];
	}

	/// Get a node from an array.
	template <typename T = Serialized *>
	typename std::enable_if<std::is_same<T, Serialized *>::value, Serialized *>::type get(size_t i)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i];
	}

	/// Load the entire array into a pair of iterators.
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type get(T i, const T &end) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		NNHardAssertEquals(m_array.size(), (size_t) std::distance(i, end), "Invalid range!");
		size_t idx = 0;
		while(i != end)
			*(i++) = m_array[idx++]->as<typename std::remove_reference<decltype(*i)>::type>();
	}

	/// Set a value in an array.
	template <typename T, typename ... Ts>
	void set(size_t i, T && value, Ts && ...values)
	{
		set(i, new Serialized(std::forward<T>(value), std::forward<Ts>(values)...));
	}

	/// Set a node in an array.
	void set(size_t i, Serialized *value)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		delete m_array[i];
		m_array[i] = value;
	}

	/// Add a value to an array.
	template <typename ... Ts>
	void add(Ts && ...values)
	{
		type(Array);
		m_array.add(new Serialized(std::forward<Ts>(values)...));
	}

	/// Add a node to an array.
	void add(Serialized *value)
	{
		type(Array);
		m_array.add(value);
	}

// MARK: Object methods

	/// Check whether the given key exists in an object.
	bool has(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object.has(key);
	}

	/// Get the type of an element in an object.
	Type type(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->type();
	}

	/// Get a numeric value from an object.
	template <typename T>
	typename std::enable_if<(std::is_arithmetic<T>::value || traits::HasLoadAndSave<T>::value) && !std::is_pointer<T>::value, T>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}

	/// Get a non-numeric value from an object.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !std::is_pointer<T>::value && !traits::HasLoadAndSave<T>::value, const T &>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}

	/// Get a non-numeric value from an object.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !std::is_pointer<T>::value && !traits::HasLoadAndSave<T>::value, T &>::type get(const std::string &key)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}

	/// Get a pointer value from an object.
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && !std::is_same<T, Serialized *>::value, T>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}

	/// Get a node from an object.
	template <typename T = Serialized *>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const Serialized *>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key];
	}

	/// Get a node from an object.
	template <typename T = Serialized *>
	typename std::enable_if<std::is_same<T, Serialized *>::value, Serialized *>::type get(const std::string &key)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key];
	}

	/// Load a value into a variable from an object.
	template <typename T>
	void get(const std::string &key, T &value) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		value = m_object[key]->as<T>();
	}

	/// Load an entire array into a pair of iterators.
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type get(const std::string &key, T i, const T &end) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		m_object[key]->get(i, end);
	}

	/// Set a value in an object.
	template <typename T, typename ... Ts>
	void set(const std::string &key, T && value, Ts && ...values)
	{
		type(Object);
		m_object.set(key, new Serialized(std::forward<T>(value), std::forward<Ts>(values)...));
	}

	/// Set a node in an object.
	void set(const std::string &key, Serialized *value)
	{
		type(Object);
		m_object.set(key, value);
	}

private:
	Type m_type;
	union
	{
		bool m_bool;
		int m_int;
		double m_float;
		std::string m_string;
		SerializedArray m_array;
		SerializedObject m_object;
	};
};

using SerializedArray = Serialized::SerializedArray;
using SerializedObject = Serialized::SerializedObject;

}

#endif
