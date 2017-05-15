#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <functional>

namespace nnlib
{

/// A wrapper for reading from files or strings to objects.
/// Can be used as an iostream with >> and <<.
/// Assumes things are deserialized in the same order and with the same types.
/// \todo Figure out how to enforce safe serialization.
class Archive
{
public:
	/// \brief A static class used to hold constructors of classes derived from a shared base.
	///
	/// This is needed for polymorphic serialization.
	template <typename Base>
	struct Mapper
	{
		typedef std::function<void*()> constructor;
		static std::unordered_map<std::string, constructor> map;
		static bool add(std::string name, constructor c)
		{
			NNAssert(map.find(name) == map.end(), "Attempted to redefine mapped class!");
			map.emplace(name, c);
			return true;
		}
	};
	
	/// A dummy class used to add derived class bindings.
	template <typename Derived>
	struct Binding
	{
		static bool bind;
	};
	
	/// \brief Create an archive that reads from a file.
	///
	/// \param filename The file to read from.
	/// \param binary Whether to open the file in binary.
	/// \return A new input archive.
	static Archive fromFile(std::string filename, bool binary = true)
	{
		int flags = std::ios::out | (binary ? std::ios::binary : 0);
		return Archive(new std::ifstream(filename.c_str(), flags), nullptr, binary, true);
	}
	
	/// \brief Create an archive that writes to a file.
	///
	/// \param filename The file to write to.
	/// \param binary Whether to open the file in binary.
	/// \return A new output archive.
	static Archive toFile(std::string filename, bool binary = true)
	{
		int flags = std::ios::out | (binary ? std::ios::binary : 0);
		return Archive(nullptr, new std::ofstream(filename.c_str(), flags), binary, true);
	}
	
	/// \brief Create an archive that reads from a string.
	///
	/// \param str The string to read from.
	/// \return A new input archive.
	static Archive fromString(std::string str)
	{
		return Archive(new std::istringstream(str), nullptr, false, true);
	}
	
	/// \brief Create an archive that writes to a string.
	///
	/// This enables the str() method for retrieving the generated string.
	/// \return A new output archive.
	static Archive toString()
	{
		return Archive(nullptr, new std::ostringstream(), false, true);
	}
	
	/// General purpose constructor.
	/// Writes to cout, reads from cin by default.
	/// \param in The input stream from which to read. May be nullptr.
	/// \param out The output stream to which to write. May be nullptr.
	/// \param binary Whether the streams are open in binary mode.
	/// \param ownsStreams Whether the archive should delete the streams when done.
	Archive(std::istream *in = &std::cin, std::ostream *out = &std::cout, bool binary = false, bool ownsStreams = false) :
		m_in(in),
		m_out(out),
		m_binary(binary),
		m_ownsStreams(ownsStreams)
	{
		if(m_in != nullptr)
		{
			size_t ver;
			*this >> ver;
			NNHardAssert(ver == serializationVersion(), "Incompatible version! Expected " + std::to_string(serializationVersion()) + ", got " + std::to_string(ver));
		}
		if(m_out != nullptr)
		{
			*this << serializationVersion();
		}
	}
	
	/// Release the streams, if owned.
	~Archive()
	{
		if(m_ownsStreams)
		{
			delete m_in;
			delete m_out;
		}
	}
	
	/// \brief Write a primative type.
	///
	/// \param x The primative to write.
	/// \return This archive, for chaining.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value, Archive>::type &operator<<(const T &x)
	{
		NNAssert(m_out != nullptr, "Archive has no output stream!");
		*m_out << x;
		if(!m_binary)
			*m_out << " ";
		return *this;
	}
	
	/// \brief Write a non-string object.
	///
	/// \param x The object to write.
	/// \return This archive, for chaining.
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value, Archive>
		::type &operator<<(const T &x)
	{
		NNAssert(m_out != nullptr, "Archive has no output stream!");
		x.save(*this);
		return *this;
	}
	
	/// \brief Write a string object.
	///
	/// \param x The string to write.
	/// \return This archive, for chaining.
	Archive &operator<<(const std::string &x)
	{
		NNAssert(m_out != nullptr, "Archive has no output stream!");
		*this << x.length();
		for(size_t i = 0, len = x.length(); i < len; ++i)
			*this << x[i];
		return *this;
	}
	
	/// \brief Read in a primative.
	///
	/// \param x The primative to read into.
	/// \return This archive, for chaining.
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value, Archive>::type &operator>>(T &x)
	{
		NNAssert(m_in != nullptr, "Archive has no input stream!");
		*m_in >> x;
		NNAssert(!m_in->fail(), "Archive failed to read primative type!");
		return *this;
	}
	
	/// \brief Read in a non-string object.
	///
	/// \param x The object to read into.
	/// \return This archive, for chaining.
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value, Archive>
		::type &operator>>(T &x)
	{
		NNAssert(m_in != nullptr, "Archive has no input stream!");
		x.load(*this);
		NNAssert(!m_in->fail(), "Archive failed to read primative type!");
		return *this;
	}
	
	/// \brief Read in a string object.
	///
	/// \param x The string to read into.
	/// \return This archive, for chaining.
	Archive &operator>>(std::string &x)
	{
		NNAssert(m_in != nullptr, "Archive has no output stream!");
		size_t len;
		*this >> len;
		x.resize(len);
		for(size_t i = 0; i < len; ++i)
			*this >> x[i];
		return *this;
	}
	
	/// \brief Read in a generic object.
	///
	/// \return A dynamically allocated object if a matching type constructor was found; null otherwise.
	template <typename Base>
	Base *read()
	{
		NNAssert(m_in != nullptr, "Archive has no input stream!");
		std::string type;
		
		auto pos = m_in->tellg();
		*this >> type;
		m_in->seekg(pos);
		
		Base *ptr = nullptr;
		auto i = Mapper<Base>::map.find(type);
		if(i != Mapper<Base>::map.end())
		{
			ptr = reinterpret_cast<Base *>(i->second());
			*this >> *ptr;
		}
		
		NNAssert(!m_in->fail(), "Archive failed while reading a generic object!");
		return ptr;
	}
	
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
	/// Versioning number for backwards compatibility.
	static size_t serializationVersion()
	{
		return 0;
	}
	
	std::istream *m_in;		///< The input stream or null.
	std::ostream *m_out;	///< The output stream or null.
	bool m_binary;			///< Whether the streams are in binary mode.
	bool m_ownsStreams;		///< Whether this archive should delete the streams.
};

/// Static initialization
template <typename T>
std::unordered_map<std::string, typename Archive::Mapper<T>::constructor> Archive::Mapper<T>::map;

/// Macro for more easily adding polymorphic types.
/*#define NNRegister()
template <>
bool Archive::Binding<Linear<>>::bind = Archive::Mapper<Module<>>::add(Linear<>::type(), [](){ return reinterpret_cast<void *>(new Linear<>()); });*/

}

#endif
