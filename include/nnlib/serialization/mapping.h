#ifndef MAPPING_H
#define MAPPING_H

#include <functional>
#include <unordered_map>
#include <string>
#include "../error.h"

namespace nnlib
{

/// A class used to hold constructors of classes derived from a shared base.
/// This is needed for polymorphic serialization.
template <typename Base>
struct Mapping
{
	typedef std::function<void*()> constructor;
	
	static std::unordered_map<std::string, constructor> &map()
	{
		static std::unordered_map<std::string, constructor> m;
		return m;
	}
	
	static std::string add(std::string name, constructor c)
	{
		NNAssert(map().find(name) == map().end(), "Attempted to redefine mapped class!");
		map().emplace(name, c);
		return name;
	}
};

}

#endif
