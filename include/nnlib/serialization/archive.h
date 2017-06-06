#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <unordered_map>
#include <string>
#include <functional>
#include <type_traits>
#include <memory>
#include "../error.h"

namespace nnlib
{

template <bool C, typename T = void>
using EnableIf = typename std::enable_if<C, T>::type;

template <typename T, typename = int>
struct HasSerialize : std::false_type
{};

template <typename T>
struct HasSerialize<T, decltype(&T::template serialize<T>, 0)> : std::true_type
{};

template <typename T, typename = int, typename = int>
struct HasLoadAndSave : std::false_type
{};

template <typename T>
struct HasLoadAndSave<T, decltype(&T::template load<T>, 0), decltype(&T::template save<T>, 0)> : std::true_type
{};

/// A class used to add derived class bindings.
/// This is needed for polymorphic serialization.
template <typename T>
struct BaseOf
{
	using type = T;
};

/// A class used to hold constructors and serializers of classes derived from a shared base.
/// This is needed for polymorphic serialization.
template <typename Base>
struct Bindings
{
	typedef std::function<Base*()> constructor;
	typedef std::function<void(Base*)> serializer;
	
	Bindings = delete;
	
	static std::unordered_map<std::string, constructor> &constructors()
	{
		static std::unordered_map<std::string, constructor> m;
		return m;
	}
	
	static std::unordered_map<std::string, serializer> &serializers()
	{
		static std::unordered_map<std::string, serializer> m;
		return m;
	}
	
	static std::unordered_map<std::string, serializer> &loaders()
	{
		static std::unordered_map<std::string, serializer> m;
		return m;
	}
	
	static std::unordered_map<std::string, serializer> &savers()
	{
		static std::unordered_map<std::string, serializer> m;
		return m;
	}
	
	static Bindings instance()
	{
		static Bindings b;
		return b;
	}
	
	template <typename Derived>
	static std::string add(std::string name)
	{
		NNAssertEquals(constructors().find(name), constructors().end(), "Attempted to rebind a bound class!");
		constructors().emplace(name, []() { return new Derived(); });
		/// \todo make the serializers... 
		/// \todo constructor is the hook, no need for std::string here
		return name;
	}
};

template <typename A>
class Archive
{
public:
	Archive(A *derived) : self(derived) {}
	
	template <typename ... Ts>
	A &operator()(Ts && ... args)
	{
		preprocess(std::forward<Ts>(args)...);
		return *self;
	}
	
private:
	template <typename T, typename ... Ts>
	void preprocess(T &&head, Ts && ... tail)
	{
		preprocess(std::forward<T>(head));
		preprocess(std::forward<Ts>(tail)...);
	}
	
	template <typename T>
	EnableIf<!HasSerialize<T>::value || !HasLoadAndSave<T>::value> preprocess(T &&arg)
	{
		self->processGeneric(arg);
	}
	
	template <typename T>
	EnableIf<HasSerialize<T>::value && HasLoadAndSave<T>::value> preprocess(T &&arg)
	{
		NNAssert(false, "Serialization failed! Type cannot have both serialize and load/save.");
	}
	
private:
	A *self;
};

template <typename A>
class InputArchive : public Archive<InputArchive<A>>
{
public:
	InputArchive(A *derived, std::istream &in) :
		Archive<InputArchive>(this),
		self(derived),
		m_in(in)
	{}
	
	template <typename T>
	EnableIf<HasSerialize<T>::value> processGeneric(T &arg)
	{
		arg.serialize(*self);
	}
	
	template <typename T>
	EnableIf<HasLoadAndSave<T>::value> processGeneric(T &arg)
	{
		arg.load(*self);
	}
	
	template <typename T>
	void processGeneric(T *&arg)
	{
		using Base = typename BaseOf<T>::type;
		
		if(arg == nullptr)
		{
			std::string type;
			
			auto pos = m_in.tellg();
			self->process(type);
			m_in.seekg(pos);
			
			auto i = Mapping<Base>::map().find(type);
			if(i != Mapping<Base>::map().end())
				arg = reinterpret_cast<Base *>(i->second());
			else
				arg = construct<Base>();
		}
		
		if(arg != nullptr)
			self->processGeneric(*arg);
		
		NNAssert(!m_in.fail(), "Archive failed while reading a generic object!");
	}
	
	template <typename T>
	EnableIf<!HasSerialize<T>::value && !HasLoadAndSave<T>::value> processGeneric(T &arg)
	{
		self->process(arg);
	}
	
private:
	template <typename T>
	EnableIf<!std::is_abstract<typename BaseOf<T>::type>::value, typename BaseOf<T>::type *> construct()
	{
		return new typename BaseOf<T>::type();
	}
	
	template <typename T>
	EnableIf<std::is_abstract<typename BaseOf<T>::type>::value, typename BaseOf<T>::type *> construct()
	{
		return nullptr;
	}
	
protected:
	A *self;
	std::istream &m_in;
};

template <typename A>
class OutputArchive : public Archive<OutputArchive<A>>
{
public:
	OutputArchive(A *derived, std::ostream &out) :
		Archive<OutputArchive>(this),
		self(derived),
		m_out(out)
	{}
	
	template <typename T>
	EnableIf<HasSerialize<T>::value> processGeneric(const T &arg)
	{
		const_cast<T &>(arg).serialize(*self);
	}
	
	template <typename T>
	EnableIf<HasLoadAndSave<T>::value> processGeneric(const T &arg)
	{
		const_cast<T &>(arg).save(*self);
	}
	
	template <typename T>
	EnableIf<!HasSerialize<T>::value && !HasLoadAndSave<T>::value> processGeneric(const T &arg)
	{
		self->process(arg);
	}
	
protected:
	A *self;
	std::ostream &m_out;
};

// Macro for more easily adding serializable and polymorphic types.
#define NNSerializable(Sub, Super)									\
	template <>														\
	struct BaseOf<Sub>												\
	{																\
		using type = Super;											\
	};																\
																	\
	std::string Binding<Sub>::name = Mapping<Super>::add(#Sub, []()	\
	{																\
		return new Sub();											\
	});

}

#endif
