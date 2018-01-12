#include "nnlib/core/error.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;
using namespace nnlib::math;

void TestMath()
{
	Tensor<NN_REAL_T> a = { 8, -6, 7, 5, 3, 0, 9, 3.14159 };
	Tensor<NN_REAL_T> b = Tensor<NN_REAL_T>({
		1, 2, 3,
		4, 5, 6,
		7, 8, 9
	}).resize(3, 3);

	NNAssertAlmostEquals(min(a), -6, 1e-12, "math::min failed!");
	NNAssertAlmostEquals(max(a), 9, 1e-12, "math::max failed!");
	NNAssertAlmostEquals(sum(a), 29.14159, 1e-12, "math::sum failed!");
	NNAssertAlmostEquals(mean(a), 3.64269875, 1e-12, "math::mean failed!");
	NNAssertAlmostEquals(variance(a), 20.964444282761, 1e-12, "math::variance failed!");
	NNAssertAlmostEquals(variance(a, true), 23.959364894584, 1e-12, "math::variance as sample failed!");

	normalize(a);
	NNAssertAlmostEquals(min(a), 0, 1e-12, "math::normalize(Tensor &) failed!");
	NNAssertAlmostEquals(max(a), 1, 1e-12, "math::normalize(Tensor &) failed!");

	normalize(a.view(2), -3.14, -1.21);
	NNAssertAlmostEquals(a(0), -1.21, 1e-12, "math::normalize(Tensor &&) failed!");
	NNAssertAlmostEquals(a(1), -3.14, 1e-12, "math::normalize(Tensor &&) failed!");

	a = { 3, 4, 5 };
	clip(a, 3.14, 4.25);
	NNAssertAlmostEquals(a(0), 3.14, 1e-12, "math::clip(Tensor &) failed!");
	NNAssertAlmostEquals(a(1), 4, 1e-12, "math::clip(Tensor &) failed!");
	NNAssertAlmostEquals(a(2), 4.25, 1e-12, "math::clip(Tensor &) failed!");

	a = { 3, 4, 5 };
	clip(a.view(2), 3.14, 4.25);
	NNAssertAlmostEquals(a(0), 3.14, 1e-12, "math::clip(Tensor &&) failed!");
	NNAssertAlmostEquals(a(1), 4, 1e-12, "math::clip(Tensor &&) failed!");
	NNAssertAlmostEquals(a(2), 5, 1e-12, "math::clip(Tensor &&) failed!");

	a = { -3, 0, 1, 4 };
	square(a);
	NNAssertAlmostEquals(a(0), 9, 1e-12, "math::square(Tensor &) failed!");
	NNAssertAlmostEquals(a(1), 0, 1e-12, "math::square(Tensor &) failed!");
	NNAssertAlmostEquals(a(2), 1, 1e-12, "math::square(Tensor &) failed!");
	NNAssertAlmostEquals(a(3), 16, 1e-12, "math::square(Tensor &) failed!");

	a = { -3, 0, 1, 4 };
	square(a.view(2));
	NNAssertAlmostEquals(a(0), 9, 1e-12, "math::square(Tensor &&) failed!");
	NNAssertAlmostEquals(a(1), 0, 1e-12, "math::square(Tensor &&) failed!");
	NNAssertAlmostEquals(a(2), 1, 1e-12, "math::square(Tensor &&) failed!");
	NNAssertAlmostEquals(a(3), 4, 1e-12, "math::square(Tensor &&) failed!");

	a.resize(3);
	sum(b, a, 0);
	NNAssertAlmostEquals(a(0), 12, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")
	NNAssertAlmostEquals(a(1), 15, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")
	NNAssertAlmostEquals(a(2), 18, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")
	sum(b, a, 1);
	NNAssertAlmostEquals(a(0), 6, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")
	NNAssertAlmostEquals(a(1), 15, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")
	NNAssertAlmostEquals(a(2), 24, 1e-12, "math::sum(const Tensor &, Tensor &, size_t) failed!")

	a = { 1, 2, 3, 4 };
	sum(b, a.view(3), 0);
	NNAssertAlmostEquals(a(0), 12, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(1), 15, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(2), 18, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(3), 4, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	sum(b, a.view(3), 1);
	NNAssertAlmostEquals(a(0), 6, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(1), 15, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(2), 24, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
	NNAssertAlmostEquals(a(3), 4, 1e-12, "math::sum(const Tensor &, Tensor &&, size_t) failed!")
}
