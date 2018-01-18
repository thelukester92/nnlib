#ifndef SERIALIZATION_SERIALIZED_TPP
#define SERIALIZATION_SERIALIZED_TPP

#include "../factory.hpp"
#include "../serialized.hpp"

namespace nnlib
{

Serialized::Serialized(Type t) :
    m_type(Null)
{
    type(t);
}

template <typename ... Ts>
Serialized::Serialized(Ts && ...values) :
    m_type(Null)
{
    set(std::forward<Ts>(values)...);
}

Serialized::Serialized(const Serialized &other) :
    m_type(Null)
{
    *this = other;
}

Serialized::Serialized(Serialized &other) :
    m_type(Null)
{
    *this = other;
}

Serialized::Serialized(Serialized &&other) :
    m_type(Null)
{
    *this = other;
}

Serialized::~Serialized()
{
    // changing the type to null will call the appropriate union destructor
    type(Null);
}

Serialized &Serialized::operator=(const Serialized &other)
{
    if(this != &other)
    {
        type(other.m_type);
        switch(m_type)
        {
        case Null:
            break;
        case Boolean:
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
            for(Serialized *&s : m_array)
                s = new Serialized(*s);
            break;
        case Object:
            m_object = other.m_object;
            for(auto p : m_object.map)
                p.second = new Serialized(p.second);
            break;
        }
    }
    return *this;
}

Serialized &Serialized::operator=(Serialized &&other)
{
    if(this != &other)
    {
        type(other.m_type);

        switch(m_type)
        {
        case Null:
            break;
        case Boolean:
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
            m_array = std::move(other.m_array);
            other.m_array.clear();
            break;
        case Object:
            m_object = std::move(other.m_object);
            other.m_object.map.clear();
            break;
        }

        other.type(Null);
    }
    return *this;
}

Serialized::Type Serialized::type() const
{
    return m_type;
}

void Serialized::type(Type type)
{
    if(type == m_type)
        return;

    if(m_type == String)
        m_string.~basic_string<char>();
    else if(m_type == Array)
    {
        for(Serialized *s : m_array)
            delete s;
        m_array.~vector<Serialized *>();
    }
    else if(m_type == Object)
    {
        for(auto p : m_object.map)
            delete p.second;
        m_object.~SerializedObject();
    }

    if(type == Type::String)
        new (&m_string) std::string;
    else if(type == Type::Array)
        new (&m_array) std::vector<Serialized *>;
    else if(type == Type::Object)
        new (&m_object) SerializedObject;

    m_type = type;
}

size_t Serialized::size() const
{
    if(m_type == Array)
        return m_array.size();
    else if(m_type == Object)
        return m_object.map.size();
    else
        return 1;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type Serialized::get() const
{
    switch(m_type)
    {
    case Null:
        return 0;
    case Boolean:
    case Integer:
        return m_int;
    case Float:
        return m_float;
    default:
        throw Error("Invalid type!");
    }
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type Serialized::get() const
{
    switch(m_type)
    {
    case Null:
        return 0;
    case Boolean:
    case Integer:
        return m_int;
    case Float:
        return m_float;
    default:
        throw Error("Invalid type!");
    }
}

template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value, T>::type Serialized::get() const
{
    switch(m_type)
    {
    case Null:
        return "null";
    case Boolean:
        return m_int ? "true" : "false";
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

template <typename T>
typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type Serialized::get() const
{
    if(m_type == Object && has("polymorphic"))
        return T(*get("data"));
    else
        return T(*this);
}

template <typename T>
typename std::enable_if<std::is_same<T, const Serialized *>::value, T>::type Serialized::get() const
{
    return this;
}

template <typename T>
typename std::enable_if<std::is_same<T, Serialized *>::value, T>::type Serialized::get()
{
    return this;
}

template <typename T>
typename std::enable_if<std::is_pointer<T>::value && std::is_abstract<typename std::remove_pointer<T>::type>::value, T>::type Serialized::get() const
{
    if(m_type == Null)
        return nullptr;
    NNHardAssert(m_type == Object && has("polymorphic"), "Expected a polymorphic type!");
    return static_cast<T>(
        Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(
            get("type")->get<std::string>(),
            *get("data")
        )
    );
}

template <typename T>
typename std::enable_if<
    std::is_pointer<T>::value && !std::is_abstract<typename std::remove_pointer<T>::type>::value &&
    !std::is_same<typename std::remove_const<typename std::remove_pointer<T>::type>::type, Serialized>::value, T
>::type Serialized::get() const
{
    if(m_type == Null)
        return nullptr;
    else if(m_type == Object && has("polymorphic"))
        return static_cast<T>(
            Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(
                get("type")->get<std::string>(),
                *get("data")
            )
        );
    else
        return new typename std::remove_pointer<T>::type (*this);
}

template <typename T>
typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type Serialized::get(T itr, const T &end) const
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertEquals(m_array.size(), (size_t) std::distance(itr, end), "Invalid range!");
    size_t idx = 0;
    while(itr != end)
    {
        *itr = m_array[idx]->get<typename std::remove_reference<decltype(*itr)>::type>();
        ++itr;
        ++idx;
    }
}

template <typename T>
typename std::enable_if<std::is_same<T, Serialized::Type>::value>::type Serialized::set(T type)
{
    this->type(type);
}

template <typename T>
typename std::enable_if<std::is_same<T, bool>::value>::type Serialized::set(T value)
{
    type(Boolean);
    m_int = value;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type Serialized::set(T value)
{
    type(Integer);
    m_int = value;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type Serialized::set(T value)
{
    type(Float);
    m_float = value;
}

template <typename T>
typename std::enable_if<std::is_convertible<T, std::string>::value && !std::is_same<T, std::nullptr_t>::value>::type Serialized::set(const T &value)
{
    type(String);
    m_string = value;
}

template <typename T>
typename std::enable_if<traits::HasSave<T>::value>::type Serialized::set(const T &value)
{
    if(Factory<typename traits::BaseOf<T>::type>::isRegistered(typeid(value)))
    {
        set("polymorphic", true);
        set("type", Factory<typename traits::BaseOf<T>::type>::derivedName(typeid(value)));
        set("data", new Serialized());
        value.save(*m_object.map.at("data"));
    }
    else
        value.save(*this);
}

template <typename T>
typename std::enable_if<std::is_pointer<T>::value && !std::is_convertible<T, std::string>::value>::type Serialized::set(const T &value)
{
    if(value == nullptr)
        type(Null);
    else
        set(*value);
}

template <typename T>
typename std::enable_if<std::is_same<T, std::nullptr_t>::value>::type Serialized::set(T value)
{
    type(Null);
}

template <typename T>
typename std::enable_if<std::is_same<T, Serialized>::value>::type Serialized::set(const T &value)
{
    *this = value;
}

template <typename T>
typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value>::type Serialized::set(T itr, const T &end)
{
    type(Array);
    m_array.clear();
    m_array.reserve(std::distance(itr, end));
    while(itr != end)
    {
        m_array.push_back(new Serialized(*itr));
        ++itr;
    }
}

template <typename T, typename ... Ts>
void Serialized::add(T && value, Ts && ...values)
{
    type(Array);
    m_array.push_back(new Serialized(std::forward<T>(value), std::forward<Ts>(values)...));
}

void Serialized::add(Serialized *value)
{
    type(Array);
    m_array.push_back(value);
}

Serialized::Type Serialized::type(size_t i) const
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    return m_array[i]->type();
}

void Serialized::type(size_t i, Type type)
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    m_array[i]->type(type);
}

size_t Serialized::size(size_t i) const
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    return m_array[i]->size();
}

template <typename T>
T Serialized::get(size_t i) const
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    return m_array[i]->get<T>();
}

template <typename T>
T Serialized::get(size_t i)
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    return m_array[i]->get<T>();
}

template <typename T>
void Serialized::get(size_t i, T itr, const T &end) const
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    m_array[i]->get(itr, end);
}

template <typename T, typename ... Ts>
void Serialized::set(size_t i, T && first, Ts && ...values)
{
    NNHardAssertEquals(m_type, Array, "Invalid type!");
    NNHardAssertLessThan(i, m_array.size(), "Invalid index!");
    m_array[i]->set(std::forward<T>(first), std::forward<Ts>(values)...);
}

bool Serialized::has(const std::string &key) const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    return m_object.map.count(key) == 1;
}

const std::vector<std::string> &Serialized::keys() const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    return m_object.keys;
}

Serialized::Type Serialized::type(const std::string &key) const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    return m_object.map.at(key)->type();
}

void Serialized::type(const std::string &key, Type type)
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    m_object.map.at(key)->type(type);
}

size_t Serialized::size(const std::string &key) const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    return m_object.map.at(key)->size();
}

template <typename T>
T Serialized::get(const std::string &key) const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    return m_object.map.at(key)->get<T>();
}

template <typename T>
T Serialized::get(const std::string &key)
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    return m_object.map.at(key)->get<T>();
}

template <typename T>
void Serialized::get(const std::string &key, T itr, const T &end) const
{
    NNHardAssertEquals(m_type, Object, "Invalid type!");
    NNHardAssert(m_object.map.count(key) == 1, "Invalid key '" + key + "'!");
    m_object.map.at(key)->get(itr, end);
}

template <typename T, typename ... Ts>
void Serialized::set(const std::string &key, T && first, Ts && ...values)
{
    type(Object);
    if(m_object.map.count(key) == 0)
    {
        m_object.map.emplace(key, new Serialized(std::forward<T>(first), std::forward<Ts>(values)...));
        m_object.keys.push_back(key);
    }
    else
        m_object.map.at(key)->set(std::forward<T>(first), std::forward<Ts>(values)...);
}

}

#endif
