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
	SerializedArray(const SerializedArray &) = delete;
	
	~SerializedArray()
	{
		for(Serialized *value : m_values)
			delete value;
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
	
	Serialized **begin()
	{
		return &m_values.front();
	}
	
	Serialized **end()
	{
		return &m_values.back();
	}
	
private:
	std::vector<Serialized *> m_values;
};

class SerializedObject
{
public:
	SerializedObject(const SerializedObject &) = delete;
	
	~SerializedObject()
	{
		for(auto &i : m_values)
			delete i.second;
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
	
	std::string *begin()
	{
		return &m_keys.begin();
	}
	
	std::string *end()
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
	
	/// No copy.
	Serialized(const Serialized &) = delete;
	
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/// Destructor; delete children if type is array or object.
	~SerializedNode()
	{
		if(m_type == Type::Array)
		{
			for(SerializedNode *n : m_array)
				delete n;
		}
		else if(m_type == Type::Object)
		{
			for(auto &nvp : m_object)
				delete nvp.second;
		}
	}
	
	/// Assignment operator.
	SerializedNode &operator=(const SerializedNode &other)
	{
		if(this != &other)
		{
			type(other.m_type);
			switch(m_type)
			{
			case Type::Null:
				break;
			case Type::Number:
				m_number = other.m_number;
				break;
			case Type::String:
				m_string = other.m_string;
				break;
			case Type::Array:
				m_array.clear();
				for(SerializedNode *n : other.m_array)
					m_array.push_back(new SerializedNode(*n));
				break;
			case Type::Object:
				m_object.clear();
				for(auto &it : other.m_object)
					m_object.emplace(it.first, new SerializedNode(*it.second));
				break;
			};
		}
		return *this;
	}
	
	/// Get the current type.
	Type type() const
	{
		return m_type;
	}
	
	/// \brief Set the current type.
	///
	/// Because some of the unioned types are classes, we have to explicitly
	/// call the constructor and destructor when switching between them.
	/// See http://en.cppreference.com/w/cpp/language/union.
	void type(Type type)
	{
		if(type == m_type)
			return;
		
		if(m_type == Type::String)
			m_string.~basic_string<char>();
		if(m_type == Type::Array)
			m_array.~Array();
		else if(m_type == Type::Object)
			m_object.~Object();
		
		if(type == Type::String)
			new (&m_string) std::string;
		if(type == Type::Array)
			new (&m_array) Array;
		else if(type == Type::Object)
			new (&m_object) Object;
		
		m_type = type;
	}
	
	/// Set a number value.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value>::type set(T value)
	{
		type(Type::Number);
		m_number = value;
	}
	
	/// Set a string value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, std::string>::value>::type set(const T &value)
	{
		type(Type::String);
		m_string = value;
	}
	
	/// Set an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, Array>::value>::type set(const T &value)
	{
		type(Type::Array);
		m_array = value;
	}
	
	/// Set an array value through a vector.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::vector<typename T::value_type>>::value && !std::is_same<T, Array>::value>::type set(const T &value)
	{
		type(Type::Array);
		for(auto &i : value)
			m_array.push_back(new SerializedNode(i));
	}
	
	/// Set an array value from a pair of iterators.
	template <typename T>
	void set(T i, const T &end)
	{
		Array arr;
		arr.reserve(std::distance(i, end));
		
		while(i != end)
		{
			arr.push_back(new SerializedNode(*i));
			++i;
		}
		
		set(arr);
	}
	
	/// Set an object value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, Object>::value>::type set(const T &value)
	{
		type(Type::Object);
		m_object = value;
	}
	
	/// Set a serializble value (through a reference).
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T &value)
	{
		bool isPolymorphic = true;
		
		try
		{
			set("type", Factory<typename traits::BaseOf<T>::type>::derivedName(typeid(value)));
		}
		catch(const Error &e)
		{
			isPolymorphic = false;
		}
		
		if(isPolymorphic)
		{
			SerializedNode *n = new SerializedNode();
			value.save(*n);
			set("value", n);
			set("polymorphic", true);
		}
		else
		{
			value.save(*this);
		}
	}
	
	/// Set a serializable value (through a pointer).
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T *value)
	{
		set(*value);
	}
	
	/// Assignment.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedNode>::value>::type set(const T &value)
	{
		*this = value;
	}
	
	/// Get a number value.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Number, "Invalid type!");
		return m_number;
	}
	
	/// Get a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::String, "Invalid type!");
		return m_string;
	}
	
	/// Get an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, Array>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Array, "Invalid type!");
		return m_array;
	}
	
	/// Get an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, Object>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		return m_object;
	}
	
	/// Get a serializable value.
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type as() const
	{
		T value;
		if(m_type == Type::Object && m_object.find("polymorphic") != m_object.end())
			value.load(get<SerializedNode>("value"));
		else
			value.load(*this);
		return value;
	}
	
	/// Get a polymorphic serializable value (through a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value, T>::type as() const
	{
		NNHardAssert(get<bool>("polymorphic"), "Cannot get pointer to a non-polymorphic type!");
		
		T value = dynamic_cast<T>(Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(get<std::string>("type")));
		NNHardAssertNotEquals(value, nullptr, "Failed to get pointer to derived class!");
		
		value->load(get<SerializedNode>("value"));
		return value;
	}
	
	/// Get identity (useful for convenience).
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedNode>::value, const T &>::type as() const
	{
		return *this;
	}
	
	/// In an array or object, get the size. Other types always return 1.
	size_t size() const
	{
		switch(m_type)
		{
		case Type::Array:
			return m_array.size();
		case Type::Object:
			return m_object.size();
		default:
			return 1;
		}
	}
	
	/// \brief In an array, append a new element.
	///
	/// If the current type is not already array, this will change the type to array.
	void append(SerializedNode *value)
	{
		type(Type::Array);
		m_array.push_back(value);
	}
	
	/// \brief In an array, append a new element.
	///
	/// If the current type is not already array, this will change the type to array.
	template <typename ... Ts>
	void append(Ts && ...values)
	{
		type(Type::Array);
		m_array.push_back(new SerializedNode(std::forward<Ts>(values)...));
	}
	
	/// \brief In an object, make a name-value pair.
	///
	/// If the current type is not already object, this will change the type to object.
	void set(const std::string &name, SerializedNode *value)
	{
		type(Type::Object);
		m_object.emplace(name, value);
	}
	
	/// \brief In an object, make a name-value pair.
	///
	/// If the current type is not already object, this will change the type to object.
	template <typename T>
	void set(const std::string &name, const T &value)
	{
		type(Type::Object);
		m_object.emplace(name, new SerializedNode(value));
	}
	
	/// \brief In an object, make an array name-value pair from a pair of iterators.
	///
	/// If the current type is not already object, this will change the type to object.
	template <typename T>
	void set(const std::string &name, T i, const T &end)
	{
		Array arr;
		arr.reserve(std::distance(i, end));
		
		while(i != end)
		{
			arr.push_back(new SerializedNode(*i));
			++i;
		}
		
		set(name, arr);
	}
	
	/// \brief In an object, get a number or serializable type from a name-value pair.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value || traits::HasLoadAndSave<T>::value || std::is_pointer<T>::value, T>::type get(const std::string &name) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		return i->second->as<T>();
	}
	
	/// \brief In an object, get a non-number, non-serializable, non-pointer type from a name-value pair.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !traits::HasLoadAndSave<T>::value && !std::is_pointer<T>::value, const T &>::type get(const std::string &name) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		return i->second->as<T>();
	}
	
	/// \brief In an object, load a value from a name-value pair into a variable.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	void get(const std::string &name, T &value) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		value = i->second->as<T>();
	}
	
	/// \brief In an object, load an array from a name-value pair into a pair of iterators.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	void get(const std::string &name, T i, const T &end) const
	{
		NNHardAssertEquals(std::distance(i, end), get<Array>(name).size(), "Invalid range!");
		for(SerializedNode *n : get<Array>(name))
		{
			*i = n->as<typename std::remove_reference<decltype(*i)>::type>();
			++i;
		}
	}
	
	/// Load an array into a pair of iterators.
	template <typename T>
	void get(T i, const T &end) const
	{
		NNHardAssertEquals(m_type, Type::Array, "Invalid type!");
		NNHardAssertEquals(std::distance(i, end), m_array.size(), "Invalid range!");
		for(SerializedNode *n : m_array)
		{
			*i = n->as<typename std::remove_reference<decltype(*i)>::type>();
			++i;
		}
	}
	
private:
	/// A tag indicating the active type.
	Type m_type;
	
	/// The actual data.
	union
	{
		double m_number;      ///< The number value. All numbers are stored as double here.
		std::string m_string; ///< The string value.
		Array m_array;        ///< The array value (a vector of SerializedNodes).
		Object m_object;      ///< The object value (a map of string -> SerializedNodes).
	};
};

}

#endif
