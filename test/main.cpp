// force debugging asserts
#ifdef OPTIMIZE
	#warning Debugging asserts have been re-enabled for testing.
	#undef OPTIMIZE
#endif

#include "core/test_error.hpp"
#include "core/test_storage.hpp"
#include "core/test_tensor.hpp"
#include "critics/test_criticsequencer.hpp"
#include "critics/test_mse.hpp"
#include "critics/test_nll.hpp"
#include "math/test_math_base.hpp"
#include "math/test_math_blas.hpp"
#include "nn/test_batchnorm.hpp"
#include "nn/test_concat.hpp"
#include "nn/test_dropconnect.hpp"
#include "nn/test_dropout.hpp"
#include "nn/test_linear.hpp"
#include "nn/test_logistic.hpp"
#include "nn/test_logsoftmax.hpp"
#include "nn/test_lstm.hpp"
#include "nn/test_relu.hpp"
#include "nn/test_sequencer.hpp"
#include "nn/test_sequential.hpp"
#include "nn/test_tanh.hpp"
#include "opt/test_adam.hpp"
#include "opt/test_nadam.hpp"
#include "opt/test_rmsprop.hpp"
#include "opt/test_sgd.hpp"
#include "serialization/test_csvserializer.hpp"
#include "serialization/test_jsonserializer.hpp"
#include "serialization/test_serialized.hpp"
#include "util/test_args.hpp"
#include "util/test_batcher.hpp"
#include "util/test_random.hpp"
using namespace nnlib;

#include <iostream>
#include <string>
#include <initializer_list>
#include <tuple>
#include <functional>
using namespace std;

#define TEST(Prefix, Class) { string("Testing ") + Prefix + #Class, Test##Class }

int main()
{
	int ret = 0;
	
	initializer_list<pair<string, function<void()>>> tests = {
		TEST("core/", Error),
		TEST("core/", Storage),
		TEST("core/", Tensor),
		TEST("critics/", CriticSequencer),
		TEST("critics/", MSE),
		TEST("critics/", NLL),
		TEST("math/", MathBase),
		TEST("math/", MathBLAS),
		TEST("nn/", BatchNorm),
		TEST("nn/", Concat),
		TEST("nn/", DropConnect),
		TEST("nn/", Dropout),
		TEST("nn/", Linear),
		TEST("nn/", Logistic),
		TEST("nn/", LogSoftMax),
		TEST("nn/", LSTM),
		TEST("nn/", ReLU),
		TEST("nn/", Sequencer),
		TEST("nn/", Sequential),
		TEST("nn/", TanH),
		TEST("opt/", Adam),
		TEST("opt/", Nadam),
		TEST("opt/", RMSProp),
		TEST("opt/", SGD),
		TEST("serialization/", CSVSerializer),
		TEST("serialization/", JSONSerializer),
		TEST("serialization/", Serialized),
		TEST("util/", Args),
		TEST("util/", Batcher),
		TEST("util/", Random)
	};
	
	try
	{
		for(auto test : tests)
		{
			cout << test.first << "..." << flush;
			test.second();
			cout << " Passed!" << endl;
		}
		
		cout << "All tests passed!" << endl;
	}
	catch(const Error &e)
	{
		cerr << endl << "An error occurred:\n\t" << e.what() << endl;
		ret = 1;
	}
	return ret;
}
