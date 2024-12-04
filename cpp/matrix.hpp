#pragma once

#include "rust/cxx.h"
#include <span>
#include <cassert>
#include <thread>

#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
class no_init {
	static_assert(std::is_fundamental<T>::value, "should be a fundamental type");
public: 
	// constructor without initialization
	no_init () noexcept {}
	// implicit conversion T → no_init<T>
	constexpr  no_init (T value) noexcept: v_{value} {}
	// implicit conversion no_init<T> → T
	constexpr  operator T () const noexcept { return v_; }
private:
	T v_;
};

template< typename T, typename Alloc = std::allocator<T> >
class default_init_allocator : public Alloc {
	using a_t = std::allocator_traits<Alloc>;
public:
	// obtain alloc<U> where U ≠ T
	template<typename U>
	struct rebind {
		using other = default_init_allocator<U,
			typename a_t::template rebind_alloc<U> >;
	};
	// make inherited ctors visible
	using Alloc::Alloc;
	// default-construct objects
	template<typename U>
	void construct (U* ptr)
    noexcept(std::is_nothrow_default_constructible<U>::value)
	{ // 'placement new':
		::new(static_cast<void*>(ptr)) U;
	}
	// construct with ctor arguments
	template<typename U, typename... Args>
	void construct (U* ptr, Args&&... args) {
		a_t::construct(
			static_cast<Alloc&>(*this),
			ptr, std::forward<Args>(args)...);
	}
};

template <typename T>
class Matrix_t {
protected:
	const size_t rows, cols;
	std::vector<T,default_init_allocator<T>> storage;
	T *ptr;

	static constexpr size_t BLOCKDIM = 64;

	struct helper_t {
		unsigned long long rows, cols;
		T data[];
	};

public:

	Matrix_t(size_t rows, size_t cols)
    : rows(rows), cols(cols),
	  storage(rows * cols),
	  ptr(storage.data())
	{}

    // Constructor
    Matrix_t(void* buffer)
    : rows(reinterpret_cast<helper_t *>(buffer)->rows),
	  cols(reinterpret_cast<helper_t *>(buffer)->cols),
	  storage(),               // non owning keeps the vector empty
	  ptr(reinterpret_cast<helper_t *>(buffer)->data)
	{
	}

	// Copy constructor
    Matrix_t(const Matrix_t& other)
    : rows(other.rows), cols(other.cols),
	  storage(other.ptr, other.rows * other.cols), // Copy the data vector
      ptr(storage.data())
	{}

	// Move constructor
	Matrix_t(Matrix_t&& other) noexcept
    : rows(other.rows), cols(other.cols),
	  storage(std::move(other.storage)),
	  ptr(storage.empty() ? other.ptr : storage.data())
	{
	}

    // Copy assignment operator. This always return an owning matrix
    Matrix_t& operator=(const Matrix_t& other)
	{
        if (this != &other) {
            storage = std::vector<T>(other.ptr, other.ptr + other.rows * other.cols); // Copy the data vector
			ptr = storage.data();
		}
        return *this;
    }

    // Access element
    double& operator()(size_t row, size_t col)
	{
        return ptr[row * cols + col];
    }

    // Const access element
    const double& operator()(size_t row, size_t col) const
	{
        return ptr[row * cols + col];
    }

	size_t size() const
	{
		return rows * cols;
	}

    void to_buffer_parallel(void* buffer) const
    {
        // This is number is from my heuristic and may be tuned
        size_t minimum_size = 8 * BLOCKDIM * BLOCKDIM;

        // We don't want to use all the threads here because this is an IO operation
        // over shared memory. * is a conservative number, so it can be improved.
        // I don't recommend to use dynamic balance here.
        const size_t n_threads = std::min(
            (size_t)8,                            // In my tests more threads don't improve io.
            size() / minimum_size // We know it is 2^n, so no need to handle remainder
        );

        // Again, we don't need to handle remainder due to 2^m
        const size_t n_per_thread = size() / n_threads;
        assert(n_per_thread % n_threads == 0);

		helper_t *helper = reinterpret_cast<helper_t *>(buffer);

		helper->rows = rows;
		helper->cols = cols;

		std::vector<std::thread> threads(n_threads);

        for (size_t i = 0; i < n_threads; ++i) {

			threads[i] = std::thread(
				[&](size_t start) {
					std::copy_n(ptr + start, n_per_thread, helper->data + start);
				},
				i * n_per_thread
			);
		}

		for (size_t i = 0; i < n_threads; i++) {
			threads[i].join();
		}
    }

	/// Serialize the Matrix_t to a payload (buffer of contiguous memory)
    void copy_to_block(Matrix_t &block, size_t row_block, size_t col_block) const
    {
		auto inbegin = ptr + row_block * cols + col_block;
		auto outbegin = block.ptr;

        // Copy from matrix to blocks
        for (size_t row = 0; row < block.rows; ++row) {

			std::copy_n(inbegin, block.cols, outbegin);
			inbegin += (cols - block.cols);
        }
    }

	/// Deserialize the matrix from a payload (buffer of contiguous memory)
    ///
    /// This uses the ptr::copy_nonoverlapping that improves
    /// vectorization copy for memory chunks
    void copy_from_block(const Matrix_t &block, size_t row_block, size_t col_block)
    {
        //assert_eq!(block.rows, block.cols, "block must be squared");
        const size_t copysize = block.cols;

		auto outbegin = ptr + row_block * cols + col_block;
		auto inbegin = block.ptr;

        // Copy from matrix to blocks
        for (size_t row = 0; row < block.rows; ++row) {

			std::copy_n(inbegin, copysize, outbegin);
			outbegin += (cols - copysize);
        }
    }

	/// Full transpose in place for small matrices
	///
    /// This function is used on the blocks to transpose inplace. As
    /// the blocks are "small" this is intended to happen in the cache.
    void transpose_small_square_inplace()
    {
		assert(rows == cols);
        for (size_t row = 0; row < rows; ++row)
			for (size_t col = 0; col < row; ++col)
				std::swap(ptr[row * cols + col], ptr[col * cols + row]);
    }

	/// Full transpose for big matrices with blocks, but without threads.
    ///
    /// This sequential version with blocks is at leat ~3x faster than
    /// the row transpose because the data is read in cache friendly
    /// order to a temporal squared blocks that fit in cache line.
    ///
    /// The transposition is performed then within the cache and
    /// written back to the main memory in cache frienly order again.
    Matrix_t transpose_big(size_t blocksize) const
    {
        Matrix_t transposed(cols, rows);
        Matrix_t block(blocksize, blocksize);

		for (size_t row_block = 0; row_block < rows; row_block += blocksize) {
			for (size_t col_block = 0; col_block < rows; col_block += blocksize) {
				copy_to_block(block, row_block, col_block);
                block.transpose_small_square_inplace();
                transposed.copy_from_block(block, col_block, row_block);
			}
		}

        return transposed;
    }

	/// Full transpose for big matrices with blocks and threads.
    /// This version uses dynamic dispatch to solve potential imbalances
    /// when the host cores have different speed
    Matrix_t transpose_parallel_dynamic(size_t blocksize) const
    {
        const size_t n_threads = std::thread::hardware_concurrency();

        Matrix_t transposed(cols, rows);

        const size_t blocks_cols = cols / blocksize;
        const size_t total_blocks = (rows / blocksize) * blocks_cols;

		std::vector<std::thread> threads;
		threads.reserve(n_threads);

        std::atomic<size_t> counter(0);

		for (size_t i = 0; i < n_threads; ++i) {

			if (i >= total_blocks) {
                break;
            }

			threads.push_back(std::thread(
				[&]() {
                    Matrix_t block(blocksize, blocksize);

                    for (size_t blockid = counter.fetch_add(1);
						 blockid < total_blocks;
						 blockid = counter.fetch_add(1)
					) {
                        size_t first_row = (blockid / blocks_cols) * blocksize;
                        size_t first_col = (blockid % blocks_cols) * blocksize;

                        copy_to_block(block, first_row, first_col);
                        block.transpose_small_square_inplace();
                        transposed.copy_from_block(&block, first_col, first_row);
					}
				}
			));
		}

		for (size_t i = 0; i < n_threads; i++) {
			threads[i].join();
		}

		return transposed;
    }

};

// template<>
// Matrix_t<double> Matrix_t<double>::transpose_big(size_t blocksize) const;


class Matrix : public Matrix_t<double> {
public:

	using Matrix_t<double>::Matrix_t;

	Matrix(Matrix_t<double> &&parent)
	: Matrix_t<double>(std::move(parent))
	{}

	void to_buffer(uint8_t* buffer) const
	{
		to_buffer_parallel(static_cast<void *>(buffer));
	}

	std::unique_ptr<Matrix> transpose() const
	{
		
		//return std::make_unique<Matrix>(transpose_parallel_dynamic(Matrix_t<double>::BLOCKDIM));
		return std::make_unique<Matrix>(transpose_big(Matrix_t<double>::BLOCKDIM));
	}
};


inline std::unique_ptr<Matrix> from_buffer(uint8_t* data)
{
	return std::make_unique<Matrix>(static_cast<void *>(data));
}
