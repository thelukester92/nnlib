#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <iostream>
#include "detail.h"

namespace nnlib
{

/// \brief Base class for input archives.
///
/// An archive, as opposed to a serializer, reads many kinds of data.
/// In particular, any class with a serialize(Archive) or load(Archive)
/// can be read into from an InputArchive.
template <typename Derived>
class InputArchive
{
public:
	InputArchive(Derived *self) : self(self) {}
	
	/// Multiple types at once.
	template <typename T>
	void operator()(T && arg)
	{
		preprocess(std::forward<T>(arg));
	}
	
	/// Multiple types at once.
	template <typename T, typename ... Ts>
	void operator()(T && arg, Ts && ... args)
	{
		preprocess(std::forward<T>(arg));
		(*this)(std::forward<Ts>(args)...);
	}
	
private:
	/// Primitives and strings.
	template <typename T>
	typename std::enable_if<!std::is_polymorphic<T>::value && !std::is_pointer<T>::value>::type preprocess(T &arg)
	{
		self->process(arg);
	}
	
	/// Polymorphic serializable objects (through pointer).
	template <typename T>
	typename std::enable_if<std::is_polymorphic<T>::value>::type preprocess(T *&arg)
	{
		using Base = typename detail::BaseOf<T>::type;
		
		std::string name;
		self->process(name);
		
		arg = static_cast<T *>(detail::Binding<Base>::construct(name));
		detail::Binding<Base>::serialize(*self, *arg);
	}
	
	/// Polymorphic serializable objects (through reference).
	template <typename T>
	typename std::enable_if<std::is_polymorphic<T>::value>::type preprocess(T &arg)
	{
		using Base = typename detail::BaseOf<T>::type;
		
		std::string name;
		self->process(name);
		
		detail::Binding<Base>::serialize(*self, arg);
	}
	
	Derived *self;
};

/// \brief Base class for output archives.
///
/// An archive, as opposed to a serializer, write many kinds of data.
/// In particular, any class with a serialize(Archive) or save(Archive)
/// can be written by an OutputArchive.
template <typename Derived>
class OutputArchive
{
public:
	OutputArchive(Derived *self) : self(self) {}
	
	/// Multiple types at once.
	template <typename T>
	void operator()(T && arg)
	{
		preprocess(std::forward<T>(arg));
	}
	
	/// Multiple types at once.
	template <typename T, typename ... Ts>
	void operator()(T && arg, Ts && ... args)
	{
		preprocess(std::forward<T>(arg));
		(*this)(std::forward<Ts>(args)...);
	}
	
private:
	/// Primitives, strings, and nonpolymorphic objects.
	template <typename T>
	typename std::enable_if<!std::is_polymorphic<T>::value && !std::is_pointer<T>::value>::type preprocess(const T &arg)
	{
		self->process(arg);
	}
	
	/// Polymorphic serializable objects (through pointer).
	template <typename T>
	typename std::enable_if<std::is_polymorphic<T>::value>::type preprocess(const T *arg)
	{
		using Base = typename detail::BaseOf<T>::type;
		(*self)(detail::Binding<Base>::bindingName(typeid(*arg)));
		detail::Binding<Base>::serialize(*self, *const_cast<T *>(arg));
	}
	
	/// Polymorphic serializable objects (through reference).
	template <typename T>
	typename std::enable_if<std::is_polymorphic<T>::value>::type preprocess(const T &arg)
	{
		using Base = typename detail::BaseOf<T>::type;
		(*self)(detail::Binding<Base>::bindingName(typeid(arg)));
		detail::Binding<Base>::serialize(*self, const_cast<T &>(arg));
	}
	
	Derived *self;
};

}

#endif
