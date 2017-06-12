#ifndef SERIALIZATION_BASIC_H
#define SERIALIZATION_BASIC_H

#include "archive.h"

namespace nnlib
{

class BasicInputArchive : public InputArchive<BasicInputArchive>
{
public:
	BasicInputArchive(std::istream &in) :
		InputArchive<BasicInputArchive>(this),
		m_in(in)
	{}
	
	/// Serialize a string.
	void process(std::string &arg)
	{
		m_in >> arg;
	}
	
	/// Serialize a primitive.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value>::type process(T &arg)
	{
		m_in >> arg;
	}
	
	/// Serialize an object.
	template <typename T>
	typename std::enable_if<detail::HasSerialize<T>::value>::type process(T &arg)
	{
		arg.serialize(*this);
	}
	
	/// Serialize an object.
	template <typename T>
	typename std::enable_if<detail::HasLoadAndSave<T>::value>::type process(T &arg)
	{
		arg.load(*this);
	}
	
private:
	std::istream &m_in;
};

class BasicOutputArchive : public OutputArchive<BasicOutputArchive>
{
public:
	BasicOutputArchive(std::ostream &out) :
		OutputArchive<BasicOutputArchive>(this),
		m_out(out)
	{}
	
	/// Serialize a string.
	void process(const std::string &arg)
	{
		m_out << arg << " ";
	}
	
	/// Serialize a primitive.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value>::type process(const T &arg)
	{
		m_out << arg << " ";
	}
	
	/// Serialize an object.
	template <typename T>
	typename std::enable_if<detail::HasSerialize<T>::value>::type process(const T &arg)
	{
		const_cast<T &>(arg).serialize(*this);
	}
	
	/// Serialize an object.
	template <typename T>
	typename std::enable_if<detail::HasLoadAndSave<T>::value>::type process(const T &arg)
	{
		arg.save(*this);
	}
	
private:
	std::ostream &m_out;
};

}

NNRegisterArchive(BasicInputArchive);
NNRegisterArchive(BasicOutputArchive);

#endif
