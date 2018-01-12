#ifndef CORE_TENSOR_HPP
#define CORE_TENSOR_HPP

#include "storage.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <functional>

namespace nnlib
{

class Serialized;

template <typename T>
class TensorIterator;

/// \brief The standard input and output type in nnlib.
///
/// A tensor can be a vector (one dimension), a matrix (two dimensions), or a higher-order tensor.
/// Tensors provide views into Storage objects, and multiple tensors can share the same Storage.
template <typename T = NN_REAL_T>
class Tensor
{
public:
	/// \brief Vectorizes a list of tensors.
	///
	/// Each tensor in the parameter becomes a subview into a single, contiguous vector.
	/// If the tensors are already vectorized, this returns a vector viewing the entire list.
	/// Otherwise, all shared connections are broken and a new vector is created.
	/// \param tensors A list of tensors to vectorize.
	/// \return The vectorized data (shared by the original tensors).
	static Tensor vectorize(const Storage<Tensor *> &tensors);

	/// \brief Concatenates a number of tensors along the given dimension.
	///
	/// Each tensor in the parameter becomes a subview into a single shared Storage.
	/// If any of the tensors shared data, the old links are broken and are no longer shared.
	/// Unlike flatten, concatenate requires that the tensors are compatible (same dimensions except for the concatinating dimension).
	/// By default, this will concatenate along the final dimension.
	/// \param tensors A list of tensors to concatenate.
	/// \return The concatenated tensor.
	static Tensor concatenate(const Storage<Tensor *> &tensors, size_t dim = (size_t) -1);

	/// Generate a vector containing a random permutation of integers in [0, n)
	static Tensor randPermutation(size_t n);

	/// Create a zero-length, one-dimensional tensor.
	Tensor();

	/// \brief Create a tensor with the given data.
	///
	/// This performs a deep copy of the data given in the parameter.
	/// The constructed tensor is one-dimensional (a vector).
	/// \param values A Storage containing the values to store in the new tensor.
	Tensor(const Storage<T> &values);

	/// \brief Create a tensor with the given data.
	///
	/// This performs a deep copy of the data given in the parameter.
	/// The constructed tensor is one-dimensional (a vector).
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor(const std::initializer_list<T> &values);

	/// \brief Create a tensor with the given shape.
	///
	/// This creates an n-dimensional tensor where n is the size of the input parameter.
	/// \param dims A Storage containing the dimension sizes for the new tensor.
	/// \note This contructor uses a dummy bool to differentiate itself from the const Storage<T> & constructor. This is important when T = size_t.
	Tensor(const Storage<size_t> &dims, bool);

	/// \brief Create a tensor with the given shape.
	///
	/// This creates an n-dimensional tensor where n is the size of the input parameter.
	/// \param dims A parameter pack containing the dimension sizes for the new tensor.
	template <typename ... Ts>
	explicit Tensor(size_t dim1, Ts... dims) :
		m_offset(0),
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize({ dim1, static_cast<size_t>(dims)... });
	}

	/// \brief Create a tensor as a view of another tensor with the same shape.
	///
	/// The new tensor shares the parameter's storage and copies the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This is not a copy constructor. It essentially performs a shallow copy.
	Tensor(Tensor &other);

	/// \brief Move constructor for a tensor.
	///
	/// The new tensor shares the parameter's storage and copies the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	Tensor(Tensor &&other);

	/// Load from a serialized node.
	Tensor(const Serialized &node);

	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values A Storage containing the values to store in the new tensor.
	Tensor &operator=(const Storage<T> &values);

	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor &operator=(const std::initializer_list<T> &values);

	/// \brief Make this tensor a view of another tensor and copy its shape.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
	Tensor &operator=(Tensor &other);

	/// \brief Move assignment for a tensor.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
	Tensor &operator=(Tensor &&other);

	/// Returns whether this tensor shares a buffer with another tensor.
	bool shared() const;

	/// Returns whether this tensor shares a buffer with a specific tensor.
	bool sharedWith(const Tensor &other) const;

	/// Returns whether this tensor shares a buffer with all the given tensors.
	bool sharedWith(const Storage<Tensor *> &tensors) const;

	/// Returns the number of tensors sharing data with this tensor, including this tensor.
	size_t sharedCount() const;

	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// If this shares data and the new size is greater or this is not contiguous,
	/// this method will break the connction and allocate a new buffer.
	/// When resizing bigger, the new values will be set to 0s.
	/// \param dims The new shape for the tensor.
	///
	/// \return The tensor, for chaining.
	Tensor &resize(const Storage<size_t> &dims);

	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dims A parameter pack containing the new shape for the tensor. Must not be empty.
	/// \return The tensor, for chaining.
	template <typename ... Ts>
	Tensor &resize(size_t dim1, Ts... dims)
	{
		return resize(Storage<size_t>{ dim1, static_cast<size_t>(dims)... });
	}

	/// \brief Resize one dimension of this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dim Which dimension to resize.
	/// \param size The new size of the given dimension.
	/// \return The tensor, for chaining.
	Tensor &resizeDim(size_t dim, size_t size);

	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	Tensor view(const Storage<size_t> &dims);

	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A parameter pack containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	template <typename ... Ts>
	Tensor view(Ts... dims)
	{
		return view({ static_cast<size_t>(dims)... });
	}

	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A Storage containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	const Tensor view(const Storage<size_t> &dims) const;

	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A parameter pack containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	template <typename ... Ts>
	const Tensor view(Ts... dims) const
	{
		Tensor t = const_cast<Tensor *>(this)->view({ static_cast<size_t>(dims)... });
		return t;
	}

	/// \brief Creates a new tensor with a copy of this tensor's data and a new shape.
	///
	/// This performs a deep copy of the data.
	/// The given shape must be compatible; that is, the resulting tensor must have as much data as this tensor.
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that with the given shape and a copy of the data in this tensor.
	Tensor reshape(const Storage<size_t> &dims) const;

	/// \brief Creates a new tensor with a copy of this tensor's data and a new shape.
	///
	/// This performs a deep copy of the data.
	/// The given shape must be compatible; that is, the resulting tensor must have as much data as this tensor.
	/// \param dims A parameter pack containing the new shape.
	/// \return A tensor that with the given shape and a copy of the data in this tensor.
	template <typename ... Ts>
	Tensor reshape(Ts... dims) const
	{
		return reshape({ static_cast<size_t>(dims)... });
	}

	/// \brief Creates a new tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with one less dimension than this tensor, essentially eliminating the given dimension.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to eliminate.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \return A tensor containing the subview.
	Tensor select(size_t dim, size_t index);

	/// \brief Creates a new constant tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with one less dimension than this tensor, essentially eliminating the given dimension.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to eliminate.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \return A constant tensor containing the subview.
	const Tensor select(size_t dim, size_t index) const;

	/// \brief Creates a new tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is narrowed to a specific range, which may be one or more slices in length.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to narrow.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \param size The length of the dimension to keep in the resulting tensor.
	/// \return A tensor containing the subview.
	Tensor narrow(size_t dim, size_t index, size_t size = 1);

	/// \brief Creates a new constant tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is narrowed to a specific range, which may be one or more slices in length.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to narrow.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \param size The length of the dimension to keep in the resulting tensor.
	/// \return A constant tensor containing the subview.
	const Tensor narrow(size_t dim, size_t index, size_t size = 1) const;

	/// \brief Creates a new tensor with a "superview" of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is expanded from 1 to the given number by repeating the slice n times.
	/// This is accomplished without copying by setting stride to 0 in the given dimension.
	/// The given dimension must already have a size of 1 in the given dimension.
	/// \param dim Which dimension to expand.
	/// \param size How long to expand the given dimension.
	/// \return A tensor containing the "superview."
	Tensor expand(size_t dim, size_t size);

	/// \brief Creates a new const tensor with a "superview" of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is expanded from 1 to the given number by repeating the slice n times.
	/// This is accomplished without copying by setting stride to 0 in the given dimension.
	/// The given dimension must already have a size of 1 in the given dimension.
	/// \param dim Which dimension to expand.
	/// \param size How long to expand the given dimension.
	/// \return A const tensor containing the "superview."
	const Tensor expand(size_t dim, size_t size) const;

	/// \brief Makes the given tensor a subview of this tensor's data.
	///
	/// The parameter tensor ends up with the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param t The tensor to use for the subview.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return This tensor, for chaining.
	Tensor &sub(Tensor &t, const std::initializer_list<const std::initializer_list<size_t>> &dims);

	/// \brief Creates a new tensor as a subview of this tensor's data.
	///
	/// The new tensor will have the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return The narrowed tensor.
	Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims);

	/// \brief Creates a new const tensor as a subview of this tensor's data.
	///
	/// The new tensor will have the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return The narrowed const tensor.
	const Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims) const;

	/// \brief Creates a new tensor as a copy this tensor.
	///
	/// This is a deep copy, and the resulting tensor will have the same shape as this tensor.
	/// The resulting tensor will also be contiguous, regardless of whether this tensor is contiguous.
	/// \return A copy of this tensor.
	Tensor copy() const;

	/// \brief Copies the data and shape from another tensor to this tensor.
	///
	/// This is a deep copy, but the tensor will not necessarily have the same shape, just the same size, as the other tensor.
	/// \param other The tensor to copy.
	/// \return This tensor, for chaining.
	Tensor &copy(const Tensor &other);

	/// \brief Swaps the data between two tensors.
	///
	/// The given tensor must have the same shape as this tensor.
	/// \param other The tensor with which to swap.
	/// \return This tensor, for chaining.
	Tensor &swap(Tensor &other);

	/// \brief Swaps the data between two tensors.
	///
	/// The given tensor must have the same shape as this tensor.
	/// \param other The tensor with which to swap.
	/// \return This tensor, for chaining.
	/// \note It is alright to use an rvalue reference here, as the temporary tensor is using persistant storage.
	Tensor &swap(Tensor &&other);

	/// \brief Creates a new tensor as a view of this tensor in which two dimensions are switched.
	///
	/// By default, this will switch the first and second dimension, which are rows and columns in a matrix.
	/// The resulting tensor has a view, not a copy, of this tensor.
	/// \param dim1 The first dimension to switch.
	/// \param dim2 The second dimension to switch.
	/// \return A tensor with a subview of this tensor but with the dimensions switched.
	Tensor transpose(size_t dim1 = 1, size_t dim2 = 0);

	/// Gets the list of dimension sizes.
	const Storage<size_t> &shape() const;

	/// Gets the list of dimension strides.
	const Storage<size_t> &strides() const;

	/// Gets the number of dimensions in this tensor.
	size_t dims() const;

	/// Gets the total number of elements in this tensor.
	size_t size() const;

	/// Gets the size of a given dimension.
	size_t size(size_t dim) const;

	/// Gets whether the tensor is contiguous in memory.
	bool contiguous() const;

	/// Makes the tensor contiguous in memory.
	Tensor &makeContiguous();

	/// \brief Gets the stride of a given dimension.
	///
	/// The stide is the distance between elements in the given dimension.
	/// If the dimension is contiguous, the stride in that dimension is 1.
	/// In a completely contiguous tensor, the stride between elements in dimension d
	/// is equal to the stride of dimension d+1 times the size of dimension d+1.
	/// \param dim The dimension from which to get stride.
	/// \return The stride of the given dimension.
	size_t stride(size_t dim) const;

	/// Sets every element in this tensor to the given value.
	Tensor &fill(const T &value);

	/// Sets every element in this tensor to 0.
	Tensor &zeros();

	/// Sets every element in this tensor to 1.
	Tensor &ones();

	/// \brief Sets every element in this tensor to a uniformly distributed random value.
	///
	/// \param from The lowest value in the uniform distribution.
	/// \param to The highest value in the uniform distribution.
	/// \return This tensor, for chaining.
	Tensor &rand(const T &from = -1, const T &to = 1);

	/// \brief Sets every element in this tensor to a normally distributed random value.
	///
	/// \param mean The mean of the normal distribution.
	/// \param stddev The standard deviation of the normal distribution.
	/// \return This tensor, for chaining.
	Tensor &randn(const T &mean = 0, const T &stddev = 1);

	/// \brief Sets every element in this tensor to a value sampled from a Bernoulli distribution (1 or 0).
	///
	/// \param p The probability of a 1.
	/// \return This tensor, for chaining.
	Tensor &bernoulli(const T &p = 0.5);

	/// \brief Sets every element in this tensor to a normally distributed random value, capped.
	///
	/// This resamples from the distribution when it finds a value too far away from the mean,
	/// which may be slow for a small threshold.
	/// \param mean The mean of the normal distribution.
	/// \param stddev The standard deviation of the normal distribution.
	/// \param cap The threshold value for the maximum allowed distance away from the mean.
	/// \return This tensor, for chaining.
	Tensor &randn(const T &mean, const T &stddev, const T &cap);

	/// \brief Multiplies this tensor by a scalar.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &scale(T alpha);

	/// \brief Adds a scalar to each element in this tensor.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &add(T alpha);

	/// Add another vector to this vector.
	Tensor &addV(const Tensor &x, T alpha = 1);

	/// Add another matrix to this matrix.
	Tensor &addM(const Tensor &A, T alpha = 1);

	/// Hadamard/elementwise/pointwise product.
	Tensor &pointwiseProduct(const Tensor &x);

	/// \brief Compute elementwise/pointwise sum (general purpose).
	///
	/// This is a general purpose function for any size of tensor.
	/// For vectors, addV is called; for matrices, addM is called.
	Tensor &add(const Tensor &x, T alpha = 1);

	/// Perform a pointwise product with the current tensor, squaring it.
	Tensor &square();

	/// \brief Sparsify the current dense tensor, dropping values with magnitude less than epsilon.
	///
	/// The output will be a matrix. The number of rows will be the number of non-zero elements;
	/// the number of columns will be D + 1 where D is the number of dimensions in the dense tensor.
	/// The first D columns in a row are the indices and the last is the value in that slot.
	/// The first row in the output will be the sizes of each dimension with one unused column.
	///
	/// For example, a truncated identity matrix of size 3x5 could be represented like this:
	///
	///     3 5 0.0   <-- size
	///     0 0 1.0
	///     1 1 1.0
	///     2 2 1.0
	Tensor sparsify(T epsilon = 1e-12);

	/// \brief Unsparsify the current sparse tensor.
	///
	/// See sparsify for an explanation of sparse tensors.
	Tensor unsparsify();

	/// \brief Apply the given function to each element in this tensor.
	///
	/// \note We may eventually split to apply(V|M) (see the add method) for acceleration.
	Tensor &apply(const std::function<void(T&)> &f);

	/// \brief Apply the given function to each element in this tensor.
	///
	/// \note We may eventually split to apply(V|M) (see the add method) for acceleration.
	const Tensor &apply(const std::function<void(const T&)> &f) const;

	/// Calculate the sum of all elements in this tensor.
	T sum() const;

	/// \brief Calculate the sum along the given dimension.
	///
	/// This reduces the number of dimensions by one, and may not be called on a 1D tensor (use sum() instead).
	/// For example, in this matrix:
	/// 	1 2 3
	/// 	4 5 6
	/// sum(0) will produce the vector `<5 7 9>`, and
	/// sum(1) will produce the vector `<6 15>`
	/// \param t The tensor to store the sum in. It must already be the appropriate shape.
	/// \param dim Which dimension to sum.
	/// \return The input tensor t, for chaining.
	Tensor &sum(Tensor &t, size_t dim) const;

	/// \brief Calculate the sum along the given dimension.
	///
	/// This reduces the number of dimensions by one, and may not be called on a 1D tensor (use sum() instead).
	/// For example, in this matrix:
	/// 	1 2 3
	/// 	4 5 6
	/// sum(0) will produce the vector `<5 7 9>`, and
	/// sum(1) will produce the vector `<6 15>`
	/// \param dim Which dimension to sum.
	/// \return The tensor containing the sum.
	Tensor sum(size_t dim) const;

	/// Calculate the mean of the elements of this tensor.
	T mean() const;

	/// Calculate the variance of the elements of this tensor.
	T variance(bool normalizeAsSample = false) const;

	/// Find the minimum element of this tensor.
	T min() const;

	/// Find the maximum element of this tensor.
	T max() const;

	/// Normalize the elements of this tensor.
	Tensor &normalize(T from = 0.0, T to = 1.0);

	/// Clip the elements of this tensor such that all elements lie in [smallest, largest]
	Tensor &clip(T smallest, T largest);

	T &at(const Storage<size_t> &indices);
	const T &at(const Storage<size_t> &indices) const;

	template <typename ... Ts>
	T &at(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}

	template <typename ... Ts>
	const T &at(Ts... indices) const
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}

	T &operator()(const Storage<size_t> &indices);
	const T &operator()(const Storage<size_t> &indices) const;

	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}

	template <typename ... Ts>
	const T &operator()(Ts... indices) const
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}

	T *ptr();
	const T *ptr() const;

	Storage<T> &data();
	const Storage<T> &data() const;

	TensorIterator<T> begin();
	TensorIterator<T> end();

	TensorIterator<const T> begin() const;
	TensorIterator<const T> end() const;

	void save(Serialized &node) const;

private:
	Storage<size_t> m_dims;					///< The length along each dimension.
	Storage<size_t> m_strides;				///< Strides between dimensions.
	size_t m_offset;						///< Offset of data for this view.
	Storage<T> *m_data;						///< The actual data.
	std::shared_ptr<Storage<T>> m_shared;	///< Wrapped around m_data for ARC.
	size_t m_size;							///< The total number of elements.
	bool m_contiguous;						///< Whether this tensor is contiguous (i.e. can be vectorized).

	/// Check whether the given list of tensors is already vectorized.
	static bool isVectorized(const Storage<Tensor *> &tensors);

	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const std::initializer_list<size_t> &indices) const;

	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const Storage<size_t> &indices) const;

	/// Recalculate and cache the size of this tensor.
	void recalculateSize();

	/// Check and cache whether this tensor is contiguous.
	void checkContiguous();
};

}

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Tensor<NN_REAL_T>;
#endif

#include "detail/tensor.tpp"
#include "detail/tensor_iterator.tpp"
#include "detail/tensor_operators.tpp"

#endif
