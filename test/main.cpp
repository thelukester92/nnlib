// force debugging asserts
#ifdef OPTIMIZE
	#warning Debugging asserts have been re-enabled for testing.
	#undef OPTIMIZE
#endif

#include "core/test_error.h"
#include "core/test_storage.h"
#include "core/test_tensor.h"
// #include "critics/test_criticsequencer.h"
// #include "critics/test_mse.h"
// #include "critics/test_nll.h"
#include "math/test_math_base.h"
#include "math/test_math_blas.h"
// #include "nn/test_batchnorm.h"
// #include "nn/test_concat.h"
// #include "nn/test_dropconnect.h"
// #include "nn/test_dropout.h"
// #include "nn/test_identity.h"
#include "nn/test_linear.h"
// #include "nn/test_logistic.h"
// #include "nn/test_logsoftmax.h"
// #include "nn/test_lstm.h"
// #include "nn/test_recurrent.h"
#include "nn/test_relu.h"
// #include "nn/test_sequencer.h"
// #include "nn/test_sequential.h"
#include "nn/test_tanh.h"
// #include "opt/test_adam.h"
// #include "opt/test_nadam.h"
// #include "opt/test_rmsprop.h"
// #include "opt/test_sgd.h"
// #include "serialization/test_csvserializer.h"
// #include "serialization/test_jsonserializer.h"
// #include "serialization/test_serialized.h"
#include "util/test_args.h"
#include "util/test_batcher.h"
#include "util/test_random.h"
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
		// TEST("critics/", CriticSequencer),
		// TEST("critics/", MSE),
		// TEST("critics/", NLL),
		TEST("math/", MathBase),
		TEST("math/", MathBLAS),
		// TEST("nn/", BatchNorm),
		// TEST("nn/", Concat),
		// TEST("nn/", DropConnect),
		// TEST("nn/", Dropout),
		// TEST("nn/", Identity),
		TEST("nn/", Linear),
		// TEST("nn/", Logistic),
		// TEST("nn/", LogSoftMax),
		// TEST("nn/", LSTM),
		// TEST("nn/", Recurrent),
		TEST("nn/", ReLU),
		// TEST("nn/", Sequencer),
		// TEST("nn/", Sequential),
		TEST("nn/", TanH),
		// TEST("opt/", Adam),
		// TEST("opt/", Nadam),
		// TEST("opt/", RMSProp),
		// TEST("opt/", SGD),
		// TEST("serialization/", CSVSerializer),
		// TEST("serialization/", JSONSerializer),
		// TEST("serialization/", Serialized),
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
