#include "../test_batcher.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/util/batcher.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Batcher)
{
    NNTestMethod(Batcher)
    {
        NNTestParams(Tensor &, Tensor &, size_t, bool)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> copied(feat, lab, 2, true);
            Batcher<T> notCopied(feat, lab, 2, false);
            NNTestEquals(copied.features().sharedWith(feat), false);
            NNTestEquals(copied.labels().sharedWith(lab), false);
            NNTestEquals(notCopied.features().sharedWith(feat), true);
            NNTestEquals(notCopied.labels().sharedWith(lab), true);
        }

        NNTestParams(Tensor &&, Tensor &&, size_t, bool)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> copied(std::move(feat), std::move(lab), 2, true);
            Batcher<T> notCopied(std::move(feat), std::move(lab), 2, false);
            NNTestEquals(copied.features().sharedWith(feat), false);
            NNTestEquals(copied.labels().sharedWith(lab), false);
            NNTestEquals(notCopied.features().sharedWith(feat), true);
            NNTestEquals(notCopied.labels().sharedWith(lab), true);
        }

        NNTestParams(const Tensor &, const Tensor &, size_t)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> feat(6, 2), lab(6, 1);
            const Tensor<T> &cFeat = feat;
            const Tensor<T> &cLab = lab;
            Batcher<T> batcher(cFeat, cLab, 2);
            NNTestEquals(batcher.features().sharedWith(cFeat), false);
            NNTestEquals(batcher.labels().sharedWith(cLab), false);
        }
    }

    NNTestMethod(batch)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 2);
            NNTestEquals(batcher.batch(), 2);
        }

        NNTestParams(size_t)
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 2);
            batcher.batch(5);
            NNTestEquals(batcher.batch(), 5);
        }
    }

    NNTestMethod(batches)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 2);
            NNTestEquals(batcher.batches(), 3);
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 1);
            Storage<bool> included(6);

            for(size_t trial = 0; trial < 10; ++trial)
            {
                batcher.reset();
                for(size_t i = 0; i < 6; ++i)
                    included[i] = false;

                for(size_t i = 0; i < 6; ++i)
                {
                    for(size_t j = 0; j < 6; ++j)
                    {
                        if(&feat(j, 0) == &batcher.features()(0, 0))
                        {
                            included[j] = true;
                            break;
                        }
                    }
                    batcher.next();
                }

                NNTestEquals(batcher.next(), false);
                for(size_t i = 0; i < 6; ++i)
                    NNTestEquals(included[i], true);
            }

            NNTestEquals(batcher.next(), false);
            NNTestEquals(batcher.next(true), true);
        }
    }

    NNTestMethod(allFeatures)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 1);
            NNTestEquals(&feat.data(), &batcher.allFeatures().data());
        }
    }

    NNTestMethod(allLabels)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            Batcher<T> batcher(feat, lab, 1);
            NNTestEquals(&lab.data(), &batcher.allLabels().data());
        }
    }
}

NNTestClassImpl(SequenceBatcher)
{
    NNTestMethod(SequenceBatcher)
    {
        NNTestParams(const Tensor &, const Tensor &, size_t, size_t)
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            NNTestEquals(batcher.features().sharedWith(feat), false);
            NNTestEquals(batcher.labels().sharedWith(lab), false);
        }
    }

    NNTestMethod(sequenceLength)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            NNTestEquals(batcher.sequenceLength(), 3);
        }

        NNTestParams(size_t)
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            batcher.sequenceLength(4);
            NNTestEquals(batcher.sequenceLength(), 4);
        }
    }

    NNTestMethod(batch)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            NNTestEquals(batcher.batch(), 1);
        }

        NNTestParams(size_t)
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            batcher.batch(4);
            NNTestEquals(batcher.batch(), 4);
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            for(size_t i = 0; i < 6; ++i)
            {
                feat(i, 0) = i;
                lab(i, 0) = i;
            }

            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            size_t maxIterations = 10000;

            for(size_t i = 0; i < 4; ++i)
            {
                for(size_t j = 0; j < maxIterations; ++j)
                {
                    if(fabs(batcher.features()(0, 0, 0) - feat(i, 0)) < 1e-12)
                    {
                        NNTestAlmostEquals(batcher.features()(0, 0, 1), feat(i, 1), 1e-12);
                        NNTestAlmostEquals(batcher.features()(1, 0, 0), feat(i + 1, 0), 1e-12);
                        NNTestAlmostEquals(batcher.features()(1, 0, 1), feat(i + 1, 1), 1e-12);
                        NNTestAlmostEquals(batcher.features()(2, 0, 0), feat(i + 2, 0), 1e-12);
                        NNTestAlmostEquals(batcher.features()(2, 0, 1), feat(i + 2, 1), 1e-12);
                        break;
                    }
                    if(j == maxIterations - 1)
                        NNTest(false);
                    batcher.reset();
                }
            }
        }
    }

    NNTestMethod(allFeatures)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            NNTestEquals(&feat.data(), &batcher.allFeatures().data());
        }
    }

    NNTestMethod(allLabels)
    {
        NNTestParams()
        {
            Tensor<T> feat(6, 2), lab(6, 1);
            SequenceBatcher<T> batcher(feat, lab, 3, 1);
            NNTestEquals(&lab.data(), &batcher.allLabels().data());
        }
    }
}
