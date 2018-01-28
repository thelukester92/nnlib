// force debugging asserts
#ifdef NN_OPT
    #warning Debugging asserts have been re-enabled for testing.
    #undef NN_OPT
#endif

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
#include "nn/test_prelu.hpp"
#include "nn/test_relu.hpp"
#include "nn/test_sequencer.hpp"
#include "nn/test_sequential.hpp"
#include "nn/test_sin.hpp"
#include "nn/test_softmax.hpp"
#include "nn/test_tanh.hpp"
#include "opt/test_adam.hpp"
#include "opt/test_nadam.hpp"
#include "opt/test_rmsprop.hpp"
#include "opt/test_sgd.hpp"
#include "serialization/test_binaryserializer.hpp"
#include "serialization/test_csvserializer.hpp"
#include "serialization/test_fileserializer.hpp"
#include "serialization/test_jsonserializer.hpp"
#include "serialization/test_parser.hpp"
#include "serialization/test_serialized.hpp"
#include "toy_problems/classification.hpp"
#include "toy_problems/timeseries.hpp"
#include "util/test_args.hpp"
#include "util/test_batcher.hpp"
#include "util/test_progress.hpp"
#include "util/test_timer.hpp"
#include <unordered_set>

#define RunTest(Class)                                         \
    if(tests.size() == 0 || tests.find(#Class) != tests.end()) \
        NNRunTest(Class);

#define RunToyProblem(Name)                                      \
    if(tests.size() == 0 || tests.find(#Name) != tests.end())    \
    {                                                            \
        std::cout << "Testing " << #Name << "..." << std::flush; \
        Toy##Name();                                             \
        std::cout << " Done!" << std::endl;                      \
    }

int main(int argc, const char **argv)
{
    if(argc > 1 && argv[1] == std::string("-v"))
        nnlib::test::Test::verbosity() = 1;
    else if(argc > 1 && argv[1] == std::string("-V"))
        nnlib::test::Test::verbosity() = 2;

    std::unordered_set<std::string> tests;
    for(int i = 1; i < argc; ++i)
    {
        if(i == 1 && (argv[1] == std::string("-v") || argv[1] == std::string("-V")))
            continue;
        tests.emplace(argv[i]);
    }

    // Core
    RunTest(Error);
    RunTest(Storage);
    RunTest(Tensor);
    RunTest(TensorIterator);
    RunTest(TensorOperators);
    RunTest(TensorUtil);

    // Critics
    RunTest(CriticSequencer);
    RunTest(MSE);
    RunTest(NLL);

    // Math
    RunTest(Algebra);
    RunTest(Math);
    RunTest(Random);

    // Neural Network Modules
    RunTest(BatchNorm);
    RunTest(Concat);
    RunTest(DropConnect);
    RunTest(Dropout);
    RunTest(ELU);
    RunTest(Identity);
    RunTest(Linear);
    RunTest(Logistic);
    RunTest(LogSoftMax);
    RunTest(LSTM);
    RunTest(PReLU);
    RunTest(ReLU);
    RunTest(Sequencer);
    RunTest(Sequential);
    RunTest(Sin);
    RunTest(SoftMax);
    RunTest(TanH);

    // Optimizers
    RunTest(Adam);
    RunTest(Nadam);
    RunTest(RMSProp);
    RunTest(SGD);

    // Serialization
    RunTest(BinarySerializer);
    RunTest(CSVSerializer);
    RunTest(FileSerializer);
    RunTest(JSONSerializer);
    RunTest(Parser);
    RunTest(Serialized);

    // Utility Classes
    RunTest(Args);
    RunTest(ArgsParser);
    RunTest(Batcher);
    RunTest(SequenceBatcher);
    RunTest(Progress);
    RunTest(Timer);

    // Toy Problems
    RunToyProblem(Classification);
    RunToyProblem(TimeSeries);

    return 0;
}
