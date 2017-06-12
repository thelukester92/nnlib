#ifndef TEST_BATCHER_H
#define TEST_BATCHER_H

#include "nnlib/util/batcher.h"
using namespace nnlib;

void TestBatcher()
{
	Tensor<> feat(3, 3), lab(3, 3);
	Batcher<> batcher(feat, lab);
	SequenceBatcher<> sBatcher(feat, lab);
}

#endif
