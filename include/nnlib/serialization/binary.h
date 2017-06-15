#ifndef SERIALIZATION_BINARY_H
#define SERIALIZATION_BINARY_H

#include "archive.h"

namespace nnlib
{

class BinaryInputArchive : public InputArchive<BinaryInputArchive>
{
public:
	BinaryInputArchive(std::istream &in) :
		InputArchive<BinaryInputArchive>(this),
		m_in(in)
	{}
	
	/// Serialize a string.
	void process(std::string &arg)
	{
		size_t n;
		process(n);
		arg.resize(n);
		m_in.read(const_cast<char *>(arg.data()), n);
	}
	
	/// Serialize a primitive.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value>::type process(T &arg)
	{
		m_in.read((char *) &arg, sizeof(T));
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

class BinaryOutputArchive : public OutputArchive<BinaryOutputArchive>
{
public:
	BinaryOutputArchive(std::ostream &out) :
		OutputArchive<BinaryOutputArchive>(this),
		m_out(out)
	{}
	
	/// Serialize a string.
	void process(const std::string &arg)
	{
		process(arg.size());
		m_out.write(arg.data(), arg.size());
	}
	
	/// Serialize a primitive.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value>::type process(const T &arg)
	{
		m_out.write((char *) &arg, sizeof(T));
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

NNRegisterArchive(BinaryInputArchive);
NNRegisterArchive(BinaryOutputArchive);

#endif
