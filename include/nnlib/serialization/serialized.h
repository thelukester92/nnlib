#ifndef SERIALIZED_H
#define SERIALIZED_H

#include <map>
#include <string>
#include <vector>
#include "factory.h"
#include "traits.h"

namespace nnlib
{

class Serialized;

class SerializedArray
{
public:
	SerializedArray(const SerializedArray &other)
	{
		*this = other;
	}
	
	~SerializedArray()
	{
		for(Serialized *value : m_values)
			delete value;
	}
	
	SerializedArray &operator=(const SerializedArray &other)
	{
		if(this != &other)
		{
			m_values.clear();
			m_values.reserve(other.m_values);
			for(Serialized *value : other.m_values)
				m_values.push_back(new Serialized(*value));
		}
		return *this;
	}
	
	size_t size() const
	{
		return m_values.size();
	}
	
	Serialized *&operator[](size_t i)
	{
		NNHardAssertLessThan(i, m_values.size(), "Index out of bounds!");
		return m_values[i];
	}
	
	void add(Serialized *value)
	{
		m_values.push_back(value);
	}
	
	const Serialized **begin() const
	{
		return &m_values.front();
	}
	
	const Serialized **end() const
	{
		return &m_values.back();
	}
	
private:
	std::vector<Serialized *> m_values;
};

class SerializedObject
{
public:
	SerializedObject(const SerializedObject &other)
	{
		*this = other;
	}
	
	~SerializedObject()
	{
		for(auto &i : m_values)
			delete i.second;
	}
	
	SerializedObject &operator=(const SerializedObject &other)
	{
		if(this != &other)
		{
			m_values.clear();
			m_keys = other.m_keys;
			for(auto &it : other.m_values))
				m_values.emplace(it.first, it.second);
		}
		return *this;
	}
	
	size_t size() const
	{
		return m_values.size();
	}
	
	Serialized *&operator[](const std::string &key)
	{
		NNHardAssertNotEquals(m_values.find(key) != m_values.end(), "No such key '" + key + "'!");
		return m_values[key];
	}
	
	void set(const std::string &key, Serialized *value)
	{
		if(m_values.find(key) != m_values.end())
		{
			delete m_values[key];
		}
		
		m_values.emplace(key, value);
		m_keys.push_back(key);
	}
	
	const std::string &*begin() const
	{
		return &m_keys.begin();
	}
	
	const std::string &*end() const
	{
		return &m_keys.end();
	}
	
private:
	std::map<std::string, Serialized *> m_values;
	std::vector<std::string> m_keys;
};

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
	Serialized(Type type = Null) :
		m_type(type)
	{}
	
	/// Create a node with the given value.
	template <typename T>
	Serialized(T && value) :
		m_type(Null)
	{
		set(std::forward<T>(value));
	}
	
	/// Copy constructor.
	Serialized(const Serialized &other) :
		m_type(Null)
	{
		*this = other;
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
		if(m_type == Array)
			m_array.~SerializedArray();
		else if(m_type == Object)
			m_object.~SerializedObject();
		
		if(type == Type::String)
			new (&m_string) std::string;
		if(type == Type::Array)
			new (&m_array) SerializedArray;
		else if(type == Type::Object)
			new (&m_object) SerializedObject;
		
		m_type = type;
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
	typename std::enable_if<std::is_integral<T>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Integer, "Invalid type!");
		return m_int;
	}
	
	/// Get a floating point value.
	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Float, "Invalid type!");
		return m_float;
	}
	
	/// Get a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, const T &>::type as() const
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
	
	/// Get an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedObject>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object;
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
	typename std::enable_if<std::is_integral<T>::value>::type set(T value)
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
	typename std::enable_if<std::is_convertible<T, std::string>::value>::type set(const T &value)
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
	
	/// Set an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedObject>::value>::type set(const T &value)
	{
		type(Object);
		m_object = value;
	}
	
// MARK: Array methods
	
	/// Get a value from an array.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value, const T &>::type get(size_t i)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}
	
	/// Get a node from an array.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const T>::type get(size_t i)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i];
	}
	
	/// Set a value in an array.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value>::type set(size_t i, const T &value)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		delete m_array[i];
		m_array[i] = new Serialized(value);
	}
	
	/// Set a node in an array.
	void set(size_t i, const Serialized *value)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		delete m_array[i];
		m_array[i] = value;
	}
	
	/// Add a value to an array.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value>::type add(const T &value)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		m_array.add(new Serialized(value));
	}
	
	/// Add a node to an array.
	void add(const Serialized *value)
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		m_array.add(value);
	}
	
// MARK: Object methods
	
	/// Get a value from an object.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value, const T &>::type get(const std::string &key)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}
	
	/// Get a node from an object.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const T>::type get(const std::string &key)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key];
	}
	
	/// Set a value in an object.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value>::type set(const std::string &key, const T &value)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		m_object.set(key, new Serialized(value));
	}
	
	/// Set a node in an object.
	void set(const std::string &key, const Serialized *value)
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
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

}

#endif
