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
#include "nn/test_identity.hpp"
#include "nn/test_linear.hpp"
#include "nn/test_logistic.hpp"
#include "nn/test_logsoftmax.hpp"
#include "nn/test_lstm.hpp"
#include "nn/test_relu.hpp"
#include "nn/test_sequencer.hpp"
#include "nn/test_sequential.hpp"
#include "nn/test_sparselinear.hpp"
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

#include "toy_problems/classification.hpp"
#include "toy_problems/timeseries.hpp"

#define UNIT_TEST(Prefix, Class) { string("Testing ") + Prefix + #Class, Test##Class }
#define TOY_PROBLEM(Name) { string("Running toy problem: ") + #Name, Toy##Name }

int main()
{
	int ret = 0;
	
	initializer_list<pair<string, function<void()>>> unit_tests = {
		UNIT_TEST("core/", Error),
		UNIT_TEST("core/", Storage),
		UNIT_TEST("core/", Tensor),
		UNIT_TEST("critics/", CriticSequencer),
		UNIT_TEST("critics/", MSE),
		UNIT_TEST("critics/", NLL),
		UNIT_TEST("math/", MathBase),
		UNIT_TEST("math/", MathBLAS),
		UNIT_TEST("nn/", BatchNorm),
		UNIT_TEST("nn/", Concat),
		UNIT_TEST("nn/", DropConnect),
		UNIT_TEST("nn/", Dropout),
		UNIT_TEST("nn/", Identity),
		UNIT_TEST("nn/", Linear),
		UNIT_TEST("nn/", Logistic),
		UNIT_TEST("nn/", LogSoftMax),
		UNIT_TEST("nn/", LSTM),
		UNIT_TEST("nn/", ReLU),
		UNIT_TEST("nn/", Sequencer),
		UNIT_TEST("nn/", Sequential),
		UNIT_TEST("nn/", SparseLinear),
		UNIT_TEST("nn/", TanH),
		UNIT_TEST("opt/", Adam),
		UNIT_TEST("opt/", Nadam),
		UNIT_TEST("opt/", RMSProp),
		UNIT_TEST("opt/", SGD),
		UNIT_TEST("serialization/", CSVSerializer),
		UNIT_TEST("serialization/", JSONSerializer),
		UNIT_TEST("serialization/", Serialized),
		UNIT_TEST("util/", Args),
		UNIT_TEST("util/", Batcher),
		UNIT_TEST("util/", Random)
	};
	
	initializer_list<pair<string, function<void()>>> toy_problems = {
		TOY_PROBLEM(Classification),
		TOY_PROBLEM(TimeSeries)
	};
	
	try
	{
		for(auto test : unit_tests)
		{
			cout << test.first << "..." << flush;
			test.second();
			cout << " Passed!" << endl;
		}
		
		for(auto test : toy_problems)
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
