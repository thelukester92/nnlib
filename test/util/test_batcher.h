#ifndef TEST_BATCHER_H
#define TEST_BATCHER_H

#include "nnlib/util/batcher.h"
using namespace nnlib;

void TestBatcher()
{
	Tensor<> feat(4, 3), lab(4, 2);
	
	{
		Batcher<> batcher(feat, lab);
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "Batcher::batch failed!");
		NNAssertEquals(batcher.batches(), 2, "Batcher::batches is incorrect!");
		batcher.next();
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 3 }), "Batcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2 }), "Batcher::labels is the wrong shape!");
	}
	
	{
		Batcher<> batcher(feat, lab, 1, true);
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "Batcher::batch with copy failed!");
		NNAssertEquals(batcher.batches(), 2, "Batcher::batches with copy is incorrect!");
		batcher.next();
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 3 }), "Batcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2 }), "Batcher::labels is the wrong shape!");
	}
	
	{
		SequenceBatcher<> batcher(feat, lab);
		batcher.sequenceLength(2);
		NNAssertEquals(batcher.sequenceLength(), 2, "SequenceBatcher::sequenceLength failed!");
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "SequenceBatcher::batch failed!");
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 2, 3 }), "SequenceBatcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2, 2 }), "SequenceBatcher::labels is the wrong shape!");
	}
}

#endif
