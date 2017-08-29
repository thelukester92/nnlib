#ifndef TEST_ARCHIVE_H
#define TEST_ARCHIVE_H

#include "nnlib/serialization/archive.h"
#include "nnlib/serialization/serialized.h"
using namespace nnlib;

void TestArchive()
{
	Serialized node;
	node.set("name", "Luke");
	node.set("age", 25);
	
	Archive::output(node);
}

#endif
