#ifndef BINDING_H
#define BINDING_H

#include <string>

namespace nnlib
{

/// A class used to add derived class bindings.
/// This is needed for polymorphic serialization.
template <typename Derived>
struct Binding
{
	static std::string name;
};

}

#endif
