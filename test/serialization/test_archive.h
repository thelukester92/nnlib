#ifndef TEST_ARCHIVE_H
#define TEST_ARCHIVE_H

#include "nnlib/serialization/archive.h"
#include "nnlib/serialization/serialized_node.h"
using namespace nnlib;

void TestArchive()
{
	SerializedNode node;
	node.set("name", "Luke");
	node.set("age", 25);
	
	Archive::output(node);
}

#endif
