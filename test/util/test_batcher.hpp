#ifndef TEST_BATCHER_H
#define TEST_BATCHER_H

#include "nnlib/util/batcher.hpp"
using namespace nnlib;

void TestBatcher()
{
	Tensor<NN_REAL_T> feat(4, 3), lab(4, 2);
	
	{
		Batcher<NN_REAL_T> batcher(feat, lab);
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "Batcher::batch failed!");
		NNAssertEquals(batcher.batches(), 2, "Batcher::batches is incorrect!");
		batcher.next();
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 3 }), "Batcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2 }), "Batcher::labels is the wrong shape!");
		NNAssertEquals(batcher.allFeatures().shape(), Storage<size_t>({ 4, 3 }), "Batcher::allFeatures is the wrong shape!");
		NNAssertEquals(batcher.allLabels().shape(), Storage<size_t>({ 4, 2 }), "Batcher::allLabels is the wrong shape!");
	}
	
	{
		Batcher<NN_REAL_T> batcher(feat, lab, 1, true);
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "Batcher::batch with copy failed!");
		NNAssertEquals(batcher.batches(), 2, "Batcher::batches with copy is incorrect!");
		batcher.next();
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 3 }), "Batcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2 }), "Batcher::labels is the wrong shape!");
		
		NNAssert(batcher.next() == false, "Batcher::next(false) failed to indicate end-of-batches!");
		NNAssert(batcher.next(true), "Batcher::next(true) failed to reset the batcher!");
	}
	
	{
		SequenceBatcher<NN_REAL_T> batcher(feat, lab);
		batcher.sequenceLength(2);
		NNAssertEquals(batcher.sequenceLength(), 2, "SequenceBatcher::sequenceLength failed!");
		batcher.batch(2);
		NNAssertEquals(batcher.batch(), 2, "SequenceBatcher::batch failed!");
		NNAssertEquals(batcher.features().shape(), Storage<size_t>({ 2, 2, 3 }), "SequenceBatcher::features is the wrong shape!");
		NNAssertEquals(batcher.labels().shape(), Storage<size_t>({ 2, 2, 2 }), "SequenceBatcher::labels is the wrong shape!");
	}
}

#endif
