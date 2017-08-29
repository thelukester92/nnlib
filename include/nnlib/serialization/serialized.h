#ifndef SERIALIZED_H
#define SERIALIZED_H

#include <map>
#include <string>
#include <vector>
#include "factory.h"
#include "traits.h"

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
	/// The value of a serialized node when type is array.
	class SerializedArray
	{
	public:
		SerializedArray() = default;
		
		SerializedArray(const SerializedArray &other)
		{
			*this = other;
		}
		
		~SerializedArray()
		{
			clear();
		}
		
		SerializedArray &operator=(const SerializedArray &other)
		{
			if(this != &other)
			{
				m_values.clear();
				m_values.reserve(other.m_values.size());
				for(Serialized *value : other.m_values)
					m_values.push_back(new Serialized(*value));
			}
			return *this;
		}
		
		size_t size() const
		{
			return m_values.size();
		}
		
		void clear()
		{
			for(Serialized *value : m_values)
				delete value;
			m_values.clear();
		}
		
		Serialized *&operator[](size_t i)
		{
			NNHardAssertLessThan(i, m_values.size(), "Index out of bounds!");
			return m_values[i];
		}
		
		const Serialized *operator[](size_t i) const
		{
			NNHardAssertLessThan(i, m_values.size(), "Index out of bounds!");
			return m_values[i];
		}
		
		void add(Serialized *value)
		{
			m_values.push_back(value);
		}
		
		Serialized * const *begin() const
		{
			return &m_values.front();
		}
		
		Serialized * const *end() const
		{
			return &m_values.back() + 1;
		}
		
	private:
		std::vector<Serialized *> m_values;
	};
	
	/// The value of a serialized node when type is object.
	class SerializedObject
	{
	public:
		SerializedObject() = default;
		
		SerializedObject(const SerializedObject &other)
		{
			*this = other;
		}
		
		~SerializedObject()
		{
			clear();
		}
		
		SerializedObject &operator=(const SerializedObject &other)
		{
			if(this != &other)
			{
				m_values.clear();
				m_keys = other.m_keys;
				for(auto &it : other.m_values)
					m_values.emplace(it.first, it.second);
			}
			return *this;
		}
		
		size_t size() const
		{
			return m_values.size();
		}
		
		void clear()
		{
			for(auto &i : m_values)
				delete i.second;
			m_values.clear();
			m_keys.clear();
		}
		
		Serialized *&operator[](const std::string &key)
		{
			NNHardAssertNotEquals(m_values.find(key), m_values.end(), "No such key '" + key + "'!");
			return m_values[key];
		}
		
		const Serialized *operator[](const std::string &key) const
		{
			NNHardAssertNotEquals(m_values.find(key), m_values.end(), "No such key '" + key + "'!");
			return m_values.at(key);
		}
		
		bool has(const std::string &key) const
		{
			return m_values.find(key) != m_values.end();
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
		
		const std::string *begin() const
		{
			return &m_keys.front();
		}
		
		const std::string *end() const
		{
			return &m_keys.back() + 1;
		}
		
	private:
		std::map<std::string, Serialized *> m_values;
		std::vector<std::string> m_keys;
	};
	
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
	template <typename ... Ts>
	Serialized(Ts && ...values) :
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
	typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, T>::type as() const
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
	
	/// Get a serializable value.
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type as() const
	{
		T value;
		
		if(m_type == Object && m_object.has("polymorphic"))
			value.load(*get<Serialized *>("data"));
		else
			value.load(*this);
		
		return value;
	}
	
	/// Get a serializable value (as a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value && traits::HasLoadAndSave<typename std::remove_pointer<T>::type>::value, T>::type as() const
	{
		NNHardAssert(m_type == Object && m_object.has("polymorphic"), "Cannot deserialize a pointer to a non-polymorphic type!");
		
		std::string type = get<std::string>("type");
		
		T value = Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(type);
		value->load(*get<Serialized *>("data"));
		
		return value;
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
	
	/// Set an array value from a pair of iterators.
	template <typename T>
	typename std::enable_if<!std::is_same<T, std::string>::value>::type set(T i, const T &end)
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
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T &value)
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
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T *value)
	{
		set(*value);
	}
	
	/// Assignment.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized>::value>::type set(const T &value)
	{
		*this = value;
	}
	
// MARK: Array methods
	
	/// Get a value from an array.
	template <typename T>
	typename std::enable_if<!std::is_same<T, Serialized *>::value, const T &>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i]->as<T>();
	}
	
	/// Get a node from an array.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const T>::type get(size_t i) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		return m_array[i];
	}
	
	/// Load the entire array into a pair of iterators.
	template <typename T>
	typename std::enable_if<!std::is_same<T, std::string>::value>::type get(T i, const T &end) const
	{
		NNHardAssertEquals(m_type, Array, "Invalid type!");
		NNHardAssertEquals(m_array.size(), std::distance(i, end), "Invalid range!");
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
		if(m_type != Array)
			type(Array);
		else
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
	
	/// Get a numeric value from an object.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value || traits::HasLoadAndSave<typename std::remove_pointer<T>::type>::value, T>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}
	
	/// Get a non-numeric value from an object.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !traits::HasLoadAndSave<typename std::remove_pointer<T>::type>::value && !std::is_same<T, Serialized *>::value, const T &>::type get(const std::string &key) const
	{
		NNHardAssertEquals(m_type, Object, "Invalid type!");
		return m_object[key]->as<T>();
	}
	
	/// Get a node from an object.
	template <typename T>
	typename std::enable_if<std::is_same<T, Serialized *>::value, const Serialized *>::type get(const std::string &key) const
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
	typename std::enable_if<!std::is_same<T, std::string>::value>::type get(const std::string &key, T i, const T &end) const
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
