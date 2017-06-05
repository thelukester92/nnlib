#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <iostream>
#include "../error.h"
#include "binding.h"
#include "mapping.h"
#include "traits.h"

namespace nnlib
{

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
	EnableIf<std::is_fundamental<typename std::remove_reference<T>::type>::value> preprocess(T &&arg)
	{
		self->process(arg);
	}
	
	template <typename T>
	EnableIf<!std::is_fundamental<typename std::remove_reference<T>::type>::value> preprocess(T &&arg)
	{
		arg.serialize(*self);
	}
	
private:
	A *self;
};

class InputArchive : public Archive<InputArchive>
{
public:
	InputArchive(std::istream &in) :
		Archive<InputArchive>(this),
		m_in(in)
	{}
	
	template <typename T>
	void process(T &arg)
	{
		m_in >> arg;
	}
	
private:
	std::istream &m_in;
};

class OutputArchive : public Archive<OutputArchive>
{
public:
	OutputArchive(std::ostream &out) :
		Archive<OutputArchive>(this),
		m_out(out)
	{}
	
	template <typename T>
	void process(const T &arg)
	{
		m_out << arg;
	}
	
private:
	std::ostream &m_out;
};






public:
	
	
	
	
	
	
	
	/// \brief Get the serialized string, if using an ostringstream.
	///
	/// \return The current serialized string.
	std::string str()
	{
		NNAssert(m_out != nullptr, "Archive has no output stream!");
		std::ostringstream *oss = dynamic_cast<std::ostringstream *>(m_out);
		NNAssert(oss != nullptr, "Cannot get a string from a non-string stream!");
		return oss->str();
	}
	
private:
	/// Versioning number for backwards (in)compatibility.
	static size_t serializationVersion()
	{
		return 0;
	}
	
	std::istream *m_in;		///< The input stream or null.
	std::ostream *m_out;	///< The output stream or null.
	bool m_binary;			///< Whether the streams are in binary mode.
	bool m_ownsStreams;		///< Whether this archive should delete the streams.
};

// Macro for more easily adding serializable and polymorphic types.
#define NNSerializable(Sub, Super)									\
	template <>														\
	std::string Binding<Sub>::name = Mapper<Super>::add(#Sub, []()	\
	{																\
		return reinterpret_cast<void *>(new Sub());					\
	});

}

#endif
