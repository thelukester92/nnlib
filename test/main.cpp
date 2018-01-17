// force debugging asserts
#ifdef OPTIMIZE
    #warning Debugging asserts have been re-enabled for testing.
    #undef OPTIMIZE
#endif

// include tests
#include "test.hpp"
#include "core/test_error.hpp"
#include "core/test_storage.hpp"
#include "core/test_tensor.hpp"
#include "core/test_tensor_iterator.hpp"
#include "core/test_tensor_operators.hpp"
#include "core/test_tensor_util.hpp"
#include "critics/test_criticsequencer.hpp"
#include "critics/test_mse.hpp"
#include "critics/test_nll.hpp"
#include "math/test_algebra.hpp"
#include "math/test_math.hpp"
#include "math/test_random.hpp"
#include "nn/test_batchnorm.hpp"
#include "nn/test_concat.hpp"
#include "nn/test_dropconnect.hpp"
#include "nn/test_dropout.hpp"
#include "nn/test_elu.hpp"
#include "nn/test_identity.hpp"
#include "nn/test_linear.hpp"
#include "nn/test_logistic.hpp"
#include "nn/test_logsoftmax.hpp"
#include "nn/test_lstm.hpp"
#include "nn/test_relu.hpp"
#include "nn/test_sequencer.hpp"
#include "nn/test_sequential.hpp"
#include "nn/test_sin.hpp"
#include "nn/test_softmax.hpp"
#include "nn/test_sparselinear.hpp"
#include "nn/test_tanh.hpp"
#include "opt/test_adam.hpp"
#include "opt/test_nadam.hpp"
#include "opt/test_rmsprop.hpp"
#include "opt/test_sgd.hpp"
#include "serialization/test_binaryserializer.hpp"
#include "serialization/test_csvserializer.hpp"
#include "serialization/test_fileserializer.hpp"
#include "serialization/test_jsonserializer.hpp"
#include "serialization/test_serialized.hpp"
#include "toy_problems/classification.hpp"
#include "toy_problems/timeseries.hpp"
#include "util/test_args.hpp"
#include "util/test_batcher.hpp"

// other includes
#include "nnlib/util/args.hpp"

int main(int argc, const char **argv)
{
    nnlib::ArgsParser args;
    args.addFlag('v', "verbose");
    args.addFlag('V', "very-verbose");
    args.parse(argc, argv);

    if(args.getFlag('v'))
        nnlib::test::Test::verbosity() = 1;
    else if(args.getFlag('V'))
        nnlib::test::Test::verbosity() = 2;

    // Core
    NNRunTest(Error);
    NNRunTest(Storage);
    NNRunTest(Tensor);
    NNRunTest(TensorIterator);
    NNRunTest(TensorOperators);
    NNRunTest(TensorUtil);

    // Critics
    NNRunTest(CriticSequencer);
    NNRunTest(MSE);
    NNRunTest(NLL);

    return 0;
}
