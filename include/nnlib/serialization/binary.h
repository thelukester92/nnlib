#ifndef SERIALIZATION_BINARY_H
#define SERIALIZATION_BINARY_H

#include "archive.h"
#include <fstream>

namespace nnlib
{

class BinaryInputArchive : public InputArchive<BinaryInputArchive>
{
public:
	BinaryInputArchive(std::string filename) :
		InputArchive<BinaryInputArchive>(this),
		m_fin(new std::ifstream(filename, std::ios::in | std::ios::binary)),
		m_in(*m_fin)
	{
		NNHardAssert(m_fin->is_open(), "Failed to open " + filename);
	}
	
	BinaryInputArchive(std::istream &in) :
		InputArchive<BinaryInputArchive>(this),
		m_fin(nullptr),
		m_in(in)
	{}
	
	~BinaryInputArchive()
	{
		if(m_fin)
		{
			m_fin->close();
			delete m_fin;
		}
	}
	
	/// Whether deserialization failed.
	virtual bool fail() override
	{
		return m_in.fail();
	}
	
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
	std::ifstream *m_fin;
	std::istream &m_in;
};

class BinaryOutputArchive : public OutputArchive<BinaryOutputArchive>
{
public:
	BinaryOutputArchive(std::string filename) :
		OutputArchive<BinaryOutputArchive>(this),
		m_fout(new std::ofstream(filename, std::ios::out | std::ios::binary)),
		m_out(*m_fout)
	{
		NNHardAssert(m_fout->is_open(), "Failed to open " + filename);
	}
	
	BinaryOutputArchive(std::ostream &out) :
		OutputArchive<BinaryOutputArchive>(this),
		m_fout(nullptr),
		m_out(out)
	{}
	
	~BinaryOutputArchive()
	{
		if(m_fout)
		{
			m_fout->close();
			delete m_fout;
		}
	}
	
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
	std::ofstream *m_fout;
	std::ostream &m_out;
};

}

NNRegisterArchive(BinaryInputArchive);
NNRegisterArchive(BinaryOutputArchive);

#endif
