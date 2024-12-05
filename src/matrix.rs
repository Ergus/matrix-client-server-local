use std::{ptr,cmp,slice};
use std::sync::{Arc, RwLock};
use std::ffi::c_void;

use rand::Rng;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};

macro_rules! implement_numeric64 {
    ($($t:ty),*) => {
        $(
            impl Numeric64 for $t {
                fn zero() -> Self { 0 }
                fn one() -> Self { 1 }
            }
        )*
    }
}

/// A trait is to enforce the exercise order. It says that the type
/// needs to be 64 bits.
pub trait Numeric64: 
    Sized + 
    std::fmt::Debug + 
    Copy +
    PartialEq + 
    PartialOrd +
    Default +
    Send +
    Sync +
    std::ops::Sub<Output = Self> +
    'static 
{
    fn zero() -> Self;
    fn one() -> Self;
}

// Implement for multiple types in one go
implement_numeric64!(i64, u64, isize, usize, i128);

/// Handle f64 individually
impl Numeric64 for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

#[derive(Debug)]
enum Storage<'a, T> {
    VecData(Vec<T>),
    SliceData(&'a [T]),
}

impl<'a, T> Storage<'a, T>
{
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        match self {
            Storage::VecData(vec) => vec.iter(),
            Storage::SliceData(slice) => slice.iter(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::VecData(vec) => vec.len(),
            Storage::SliceData(slice) => slice.len(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            Storage::VecData(vec) => vec.as_ptr(),
            Storage::SliceData(slice) => slice.as_ptr(),
        }
    }

    pub fn is_owner(&self) -> bool {
        match self {
            Storage::VecData(_) => true,
            Storage::SliceData(_) => false,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Storage::VecData(vec) => vec.as_mut_ptr(),
            Storage::SliceData(slice) => {
                // Safety note: This is only safe if the slice is not
                // empty and the caller ensures no mutation of the
                // original slice
                if slice.is_empty() {
                    std::ptr::null_mut()
                } else {
                    slice.as_ptr() as *mut T
                }
            }
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Storage::VecData(vec) => vec.as_slice(),
            Storage::SliceData(slice) => slice
        }
    }

    // pub fn get(&self, idx: usize) -> T
    // {
    //     match self {
    //         Storage::VecData(vec) => vec[idx],
    //         Storage::SliceData(slice) => slice[idx]
    //     }
    // }
}

/// Operator ==
impl<T: std::cmp::PartialEq> PartialEq<Storage<'_, T>> for Storage<'_, T> {
    fn eq(&self, other: &Storage<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[derive(Debug, Clone)]
pub struct Matrix<'a, T> {
    /// Number of rows
    rows: usize,

    /// Number of columns
    cols: usize,

    /// The matrix data stored in an Arc
    data: Arc<RwLock<Storage<'a, T>>>,
}


/// A matrix class of 64 bits numbers.
///
/// The main purpose of this si to simplify and encapsulate the
/// transposition matrix operations
///
/// The class includes multiple transpositions functions specialized
/// for bigger and smaller dimensions.
///
/// The main function to use in the final application may be the
/// parallel version.
impl<T> Matrix<'_, T>
where
    T: Numeric64, Standard: Distribution<T>
{
    /// Default block dimension to use for temporal buffers.
    const BLOCKDIM: usize = 64;  // This si a simple empiric value, we may tune it.

    /// Basic constructor to create an empty matrix
    pub fn new(rows: usize, cols: usize) -> Self
    {
        let vec = vec![T::default(); rows * cols];

        Self {
            rows,
            cols,
            data: Arc::new(RwLock::new(Storage::VecData(vec))),
        }
    }

    /// Constructor to generate the matrix based on an iteration function.
    /// This is specially useful for the tests
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut vec = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                vec.push(f(i, j));
            }
        }

        Self {
            rows,
            cols,
            data: Arc::new(RwLock::new(Storage::VecData(vec)))
        }
    }

    /// Function to generate the random matrices.
    /// This is the function used in the client
    pub fn random(rows: usize, cols: usize) -> Self
    {
        let mut rng = rand::thread_rng();

        let vec = (0..rows * cols).map(|_| rng.gen()).collect();

        Self {
            rows,
            cols,
            data: Arc::new(RwLock::new(Storage::VecData(vec)))
        }
    }

    /// Copy the matrix from a memory buffer. Generally used to copy
    /// from shared memory
    pub fn from_buffer(buffer: *mut c_void) -> Self
    {
        let rows: usize = unsafe { *(buffer as *const usize) };
        let cols: usize = unsafe { *(buffer.add(8) as *const usize) };

        let slice: &[T] = unsafe {
            slice::from_raw_parts_mut(buffer.add(16) as *mut T, rows * cols)
        };

        Self {
            rows,
            cols,
            data: Arc::new(RwLock::new(Storage::SliceData(slice)))
        }
    }

    pub fn validate(&self) -> bool
    {
        {
            let rguard = self.data.read().unwrap();

            self.rows > 16
                && self.cols >= 16
                && rguard.len() == self.rows * self.cols
        }
    }

    /// Very primitive serialization function. Generally used to copy
    /// To shared memory
    fn to_buffer_seq(&self, buffer: *mut c_void)
    {
        unsafe {
            let rguard = self.data.read().unwrap();

            *(buffer as *mut usize) = self.rows;
            *(buffer.add(size_of::<usize>()) as *mut usize) = self.cols;

            ptr::copy_nonoverlapping(
                rguard.as_ptr(),
                buffer.byte_add(2 * size_of::<usize>()) as *mut T,
                self.rows * self.cols
            );
        }
    }

    /// Parallel serialization function. Generally used to copy
    /// To shared memory.
    ///
    /// The function sets a minimum size to move in parallel because
    /// the overhead of creating threads may cost more than
    /// transferring small chunks.
    ///
    /// The limit at the moment is 8 blocks, but this parameter is
    /// heuristic (almost arbitrary)
    fn to_buffer_parallel(&self, buffer: *mut c_void)
    {
        // This is number is from my heuristic and may be tuned
        let minimum_size: usize = 8 * Self::BLOCKDIM * Self::BLOCKDIM;

        // We don't want to use all the threads here because this is an IO operation
        // over shared memory. * is a conservative number, so it can be improved.
        // I don't recommend to use dynamic balance here.
        let n_threads = std::cmp::min(
            8,                            // In my tests more threads don't improve io.
            self.datalen() / minimum_size // We know it is 2^n, so no need to handle remainder
        );

        let rguard = self.data.read().unwrap();
        let base_ptr = rguard.as_ptr() as *const T;

        // Again, we don't need to handle remainder due to 2^m
        let n_per_thread = self.datalen() / n_threads;
        debug_assert_eq!(n_per_thread % n_threads, 0, "Dimension error in parallel out");

        // Update header and get payload start
        let wptr = unsafe {
            *(buffer as *mut usize) = self.rows;
            *(buffer.add(size_of::<usize>()) as *mut usize) = self.cols;
            buffer.add(2 * size_of::<usize>()) as *mut T
        };

        std::thread::scope(|s| {

            for i in 0..n_threads {

                unsafe {
                    let start_thread = i * n_per_thread;
                    let thread_rptr = AtomicPtr::new(base_ptr.add(start_thread) as *mut T);
                    let thread_wptr = AtomicPtr::new(wptr.add(start_thread) as *mut T);

                    let _ = s.spawn(move || {

                        ptr::copy_nonoverlapping(
                            thread_rptr.load(Ordering::Relaxed),
                            thread_wptr.load(Ordering::Relaxed),
                            n_per_thread
                        );
                    });
                }
            }
        })
    }

    /// Serialization function choosed to imprve IO
    ///
    /// This function establishes a minimum size of 8 blocks to execute
    /// in parallel.
    /// As we know that the matrices come as powers of 2 => 2^(m+n)
    /// Then any number above will be also a factor of the previous,
    /// meaning that the threads will execute balanced.
    pub fn to_buffer(&self, buffer: *mut c_void)
    {
        // I choose heuristically 8 blockdims
        let minimum_size: usize = 8 * Self::BLOCKDIM * Self::BLOCKDIM;

        if self.datalen() < minimum_size {
            self.to_buffer_seq(buffer)
        } else {
            self.to_buffer_parallel(buffer)
        }
    }

    /// Get Rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get Cols
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the total number of elements in the matrix.
    /// Just syntax sugar.
    pub fn datalen(&self) -> usize {
        self.cols * self.rows
    }

    /// To get the gcd we only use is the min because we assume 2^n x
    /// 2^m matrices. Otherwise we need to implement the gcd code.
    fn gcd(&self) -> usize {
        cmp::min(self.rows, self.cols)
    }

    /// This function returns the total sizze required to serialize
    /// the Matrix in a payload (buffer of contiguous memory)
    pub fn payload_size(&self) -> usize
    {
        2 * size_of::<usize>() + self.datalen() * size_of::<f64>()
    }

    pub fn all(&mut self, f: impl FnMut(&T)->bool) -> bool
    {
        let rguard = self.data.read().unwrap();

        rguard.iter().all(f)
    }

    /// Serialize the Matrix to a payload (buffer of contiguous memory)
    ///
    /// This uses the ptr::copy_nonoverlapping that improves
    /// vectorization copy for memory chunks.
    fn copy_to_block(&self, block: &mut Matrix<T>, row_block: usize, col_block: usize)
    {
        //assert_eq!(block.rows, block.cols, "block must be squared");

        let row_end: usize = (row_block + block.rows()).min(self.rows);
        let col_end: usize = (col_block + block.cols()).min(self.cols);
        let copysize: usize = col_end - col_block;

        let src: *const T = self.data.read().unwrap().as_ptr();
        let dst: *mut T = block.data.write().unwrap().as_mut_ptr();

        let mut startdst: usize = 0;

        // Copy from matrix to blocks
        for row in row_block..row_end {

            unsafe {
                // Efficient vectorized copy (~memcpy)
                ptr::copy_nonoverlapping(
                    src.add(row * self.cols + col_block),
                    dst.add(startdst),
                    copysize);
            }

            startdst += block.cols();
        }
    }

    /// Deserialize the matrix from a payload (buffer of contiguous memory)
    ///
    /// This uses the ptr::copy_nonoverlapping that improves
    /// vectorization copy for memory chunks
    fn copy_from_block(&mut self, block: &Matrix<T>, row_block: usize, col_block: usize)
    {
        assert_eq!(block.rows, block.cols, "Block must be squared");
        assert!(block.rows <= self.gcd(), "Block dim must be <= gcd");
        assert!(self.gcd() % block.rows == 0  , "Block must be a divisor of gcd");

        let row_end = row_block + block.rows();
        let col_end = col_block + block.cols();

        assert!(row_end <= self.rows, "Rows overflow coping from block");
        assert!(col_end <= self.cols, "Columns overflow coping from block");

        let copysize: usize = col_end - col_block;

        let src: *mut T = self.data.write().unwrap().as_mut_ptr();
        let dst: *const T = block.data.read().unwrap().as_ptr();

        let mut startdst: usize = 0;

        // Copy from matrix to blocks
        for row in row_block..row_end {

            unsafe {
                // Efficient vectorized copy (~memcpy)
                ptr::copy_nonoverlapping(
                    dst.add(startdst),
                    src.add(row * self.cols + col_block),
                    copysize);
            }

            startdst += block.cols();
        }
    }

    /// Full transpose in place for small matrices

    /// This function is used on the blocks to transpose inplace. As
    /// the blocks are "small" this is intended to happen in the cache.
    fn transpose_small_square_inplace(&mut self)
    {
        assert_eq!(self.rows, self.cols, "Small transpose must be squared");

        let mut wguard = self.data.write().unwrap();

        unsafe {
            let slice = slice::from_raw_parts_mut(wguard.as_mut_ptr(), wguard.len());

            for row in 0..self.rows {
                for col in 0..row {

                    let tmp: T = slice[col * self.rows + row].clone();
                    slice[col * self.rows + row] = slice[row * self.cols + col].clone();
                    slice[row * self.cols + col] = tmp;
                }
            }
        }
    }

    /// Full transpose for small matrices without blocks.
    pub fn transpose_small_rectangle(&self) -> Matrix<T>
    {
        assert!(self.rows <= 64, "Small rectangle tranpose rows must not exceed 64");
        assert!(self.cols <= 64, "Small rectangle tranpose rows must not exceed 64");

        let transpose = Matrix::<T>::new(self.cols, self.rows);

        unsafe {
            let rguard = self.data.read().unwrap();
            let mut twguard = transpose.data.write().unwrap();

            let rslice = slice::from_raw_parts(rguard.as_ptr(), rguard.len());
            let wslice = slice::from_raw_parts_mut(twguard.as_mut_ptr(), rguard.len());

            for row in 0..self.rows {
                for col in 0..self.cols {
                    wslice[col * self.rows + row] = rslice[row * self.cols + col].clone();
                }
            }
        }

        transpose
    }

    /// Full transpose for big matrices with blocks, but without threads.
    ///
    /// This sequential version with blocks is at leat ~3x faster than
    /// the row transpose because the data is read in cache friendly
    /// order to a temporal squared blocks that fit in cache line.
    ///
    /// The transposition is performed then within the cache and
    /// written back to the main memory in cache frienly order again.
    pub fn transpose_big(&self, blocksize: usize) -> Matrix<T>
    {
        let mut transposed = Matrix::<T>::new(self.cols, self.rows);

        let mut block = Matrix::<T>::new(blocksize, blocksize);

        for row_block in (0..self.rows).step_by(blocksize) {
            for col_block in (0..self.cols).step_by(blocksize) {
                self.copy_to_block(&mut block, row_block, col_block);
                block.transpose_small_square_inplace();
                transposed.copy_from_block(&block, col_block, row_block);
            }
        }

        transposed
    }

    /// Full transpose for big matrices with blocks and threads.
    /// This version user fair static dispatch 
    pub fn transpose_parallel_static(&self, blocksize: usize) -> Matrix<T>
    {
        // This si not the best approach for this because modern codes
        // have different speed which implies that using all the cores
        // at the time with similar chunk sizes implicitly introduces
        // load imbalance. But this is a 3 days job, no time for more
        // (maybe)
        let n_threads = std::thread::available_parallelism().unwrap().get();

        let transposed = Matrix::<T>::new(self.cols, self.rows);

        let blocks_cols = self.cols / blocksize;
        let total_blocks = (self.rows / blocksize) * blocks_cols;

        let blocks_per_thread = total_blocks / n_threads;
        let blocks_rest = total_blocks % n_threads;  // This is likely to be zero

        std::thread::scope(|s| {

            for i in 0..n_threads {

                let nblocks_thread = blocks_per_thread + ((i < blocks_rest) as usize);

                // This is for the case when there are less blocks than available cores
                if nblocks_thread == 0 {
                    break;
                }

                let cself = self.clone();
                let mut ctran = transposed.clone();

                let _ = s.spawn(move || {

                    let mut block = Matrix::<T>::new(blocksize, blocksize);

                    let first_block_thread = i * blocks_per_thread + cmp::min(i, blocks_rest);

                    for blockid in first_block_thread..first_block_thread + nblocks_thread {

                        let first_row = (blockid / blocks_cols) * blocksize;
                        let first_col = (blockid % blocks_cols) * blocksize;

                        cself.copy_to_block(&mut block, first_row, first_col);
                        block.transpose_small_square_inplace();
                        ctran.copy_from_block(&block, first_col, first_row);
                    }
                });
            }
        });


        transposed
    }

    /// Full transpose for big matrices with blocks and threads.
    /// This version uses dynamic dispatch to solve potential imbalances
    /// when the host cores have different speed
    pub fn transpose_parallel_dynamic(&self, blocksize: usize) -> Matrix<T>
    {
        let n_threads
            = std::thread::available_parallelism().unwrap().get();

        let transposed = Matrix::<T>::new(self.cols, self.rows);

        let blocks_cols = self.cols / blocksize;
        let total_blocks = (self.rows / blocksize) * blocks_cols;

        let counter = AtomicUsize::new(0);

        std::thread::scope(|s| {

            for i in 0..n_threads {

                // This is for the case when there are less blocks than available cores
                if i >= total_blocks {
                    break;
                }

                let cself = self.clone();
                let mut ctran = transposed.clone();
                let counter_ref = &counter;

                let _ = s.spawn(move || {

                    let mut block = Matrix::<T>::new(blocksize, blocksize);

                    loop {
                        let blockid = counter_ref.fetch_add(1, Ordering::SeqCst);
                        if blockid >= total_blocks {
                            break;
                        }

                        let first_row = (blockid / blocks_cols) * blocksize;
                        let first_col = (blockid % blocks_cols) * blocksize;

                        cself.copy_to_block(&mut block, first_row, first_col);
                        block.transpose_small_square_inplace();
                        ctran.copy_from_block(&block, first_col, first_row);
                    }
                });
            }
        });


        transposed
    }


    /// Full transpose for big matrices with blocks and threads.
    /// This version uses dynamic dispatch to solve potential imbalances
    /// when the host cores have different speed
    pub fn transpose_parallel_square_inplace(&mut self, blocksize: usize)
    {
        assert_eq!(self.cols, self.rows, "Inplace transpose is only for squared matrices.");

        let n_threads
            = std::thread::available_parallelism().unwrap().get();

        let blocks_cols = self.cols / blocksize;

        let total_blocks = blocks_cols * (blocks_cols + 1) / 2;

        let counter = AtomicUsize::new(0);

        let getidx = || -> Option<(usize, usize)>{
            let k = counter.fetch_add(1, Ordering::SeqCst);
            if k < total_blocks {
                let x = ((((8 * k + 7) as f64).sqrt() - 1.0) / 2.0).ceil() as usize - 1;
                let y = k - (x)*(x + 1)/2;

                return Some((x, y));
            }
            None
        };


        std::thread::scope(|s| {

            for i in 0..n_threads {

                // This is for the case when there are less blocks than available cores
                if i >= total_blocks {
                    break;
                }

                let mut cself = self.clone();

                let _ = s.spawn(move || {

                    let mut block1 = Matrix::<T>::new(blocksize, blocksize);
                    let mut block2 = Matrix::<T>::new(blocksize, blocksize);

                    loop {
                        match getidx() {
                            Some((row, col)) => {
                                let first_row = row * blocksize;
                                let first_col = col * blocksize;

                                cself.copy_to_block(&mut block1, first_row, first_col);
                                block1.transpose_small_square_inplace();

                                if row != col {
                                    cself.copy_to_block(&mut block2, first_col, first_row);
                                    block2.transpose_small_square_inplace();
                                }

                                cself.copy_from_block(&block1, first_col, first_row);

                                if row != col {
                                    cself.copy_from_block(&block2, first_row, first_col);
                                }
                            },
                            None => break
                        }
                    }
                });
            }
        });
    }


    pub fn transpose(&self) -> Matrix<T>
    {
        if self.cols * self.rows < Self::BLOCKDIM * Self::BLOCKDIM {
            return self.transpose_small_rectangle();
        }

        let blockdim = *[self.cols, self.rows, Self::BLOCKDIM].iter().min().unwrap();
        self.transpose_parallel_dynamic(blockdim)
    }



    /// This is intended to become the main function to use in the
    /// server code.
    ///
    /// The function uses sequential code no blocking when the total
    /// number of elements in the matrix is smaller than the prefered
    /// block dimension (BLOCKDIM).
    ///
    /// When some of the dimension is smaller than the BLOCKDIM, but
    /// the total matrix is bigger than BLOCKDIM x BLOCKDIM, we use
    /// that dimension value as blockdim.
    ///
    /// Otherwise we use BLOCKDIM x BLOCKDIM
    /// BLOCKDIM = 64 by default (hardcoded)
    pub fn transpose_inplace(&mut self)
    {
        if self.cols * self.rows < Self::BLOCKDIM * Self::BLOCKDIM {
            let transposed = self.transpose_small_rectangle();
            transposed.to_buffer(self.data.write().unwrap().as_mut_ptr() as *mut c_void);
            return;
        }

        let blockdim = *[self.cols, self.rows, Self::BLOCKDIM].iter().min().unwrap();

        // Squared matrices can be transposed in place because they
        // are not reshaped.  so no block reshape takes place and we
        // can access two blocks per thread on every call and
        // transpose them in-place "simultaneously".
        // Doing this avoids creating a temporal transposed matrix which
        // avoids extra memory and al the io are more local, almost
        // duplicating the throughput 
        if self.rows == self.cols {
            return self.transpose_parallel_square_inplace(blockdim);
        }

        // Sadly, when the matrix is rectangular we cannot use the
        // same optimization than above. So we create a temporal
        // transpose on memory and copy back to the original buffer.
        let transposed = self.transpose_parallel_dynamic(blockdim);

        if self.data.read().unwrap().is_owner() {
            unsafe {
                let mut wlk = self.data.write().unwrap();
                // We need to pad back 128 bits to set rows and cols,
                // because the data (when shared) stored only the
                // vector as a slice.
                let ptr = wlk.as_mut_ptr().sub(2) as *mut c_void;
                transposed.to_buffer(ptr);
            }
        } else {
            self.rows = transposed.rows;
            self.cols = transposed.cols;
            self.data = transposed.data;
        }

    }

    /// Get a matrix value using copy.
    pub fn get(&self, row: usize, col: usize) -> T
    {
        self.data.read().unwrap().as_slice()[row * self.cols + col]
    }

    /// Substract two matrices and obtain another matrix.
    ///
    /// We didn't implement the trait sub because the prototype is not
    /// good enough for efficiency.
    ///
    /// # Purpose
    /// This function is used in debug mode in the client to check
    /// differences in case of error.
    pub fn sub(&self, other: &Matrix<T>) -> Matrix<T> {

        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let rguard1 = self.data.read().unwrap();
        let rguard2 = other.data.read().unwrap();

        let slice1 = rguard1.as_slice();
        let slice2 = rguard2.as_slice();

        Matrix::<T>::from_fn(
            self.rows,
            self.cols,
            |i, j| slice1[i * self.cols + j] - slice2[i * self.cols + j]
        )
    }
}

/// Operator ==
impl<T: std::cmp::PartialEq> PartialEq<Matrix<'_, T>> for Matrix<'_, T> {

    fn eq(&self, other: &Matrix<T>) -> bool {
        let rguard1 = self.data.read().unwrap();
        let rguard2 = other.data.read().unwrap();

        return self.rows == other.rows
            && self.cols == other.cols
            && *rguard1 == *rguard2;
    }
}


/// Helper for print
impl<T: std::fmt::Debug> std::fmt::Display for Matrix<'_, T> {

    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        let rguard = self.data.read().unwrap();
        let gslice = rguard.as_slice();

        for i in 0..self.rows {
            let slice = &gslice[i * self.cols.. (i + 1) * self.cols];

            write!(f, "{:?}\n", slice)?;  // Format each row and move to the next line
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    use std::alloc::{alloc, dealloc, Layout};
    use std::ffi::c_void;

    #[test]
    fn test_matrix_constructor()
    {
        let matrix = Matrix::<i64>::new(3, 4);

        // Verify dimensions
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.datalen(), 4 * 3);

        // Verify all elements are zero
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(matrix.get(i, j), 0);
            }
        }
    }

    #[test]
    fn test_matrix_debug()
    {
        let matrix = Matrix::from_fn(2, 2, |i, j| i * j);
        let debug_output = format!("{:?}", matrix);

        // Ensure debug string is as expected
        assert!(debug_output.contains("Matrix"));
        assert!(debug_output.contains("data:"));
    }

    #[test]
    fn test_matrix_from_buffer()
    {
        let rows = 512;
        let cols = 1024;

        let nelems = rows * cols;

        // Calculate the total size: 2 u64s for rows and cols, then the f64 data
        let data_layout = Layout::array::<f64>(2 + nelems).expect("Layout creation failed");

        unsafe {
            let ptr = alloc(data_layout) as *mut c_void;

            if ptr.is_null() {
                panic!("Memory allocation failed!");
            }


            let hdr_ptr = ptr as *mut u64;
            std::ptr::write(hdr_ptr, rows as u64);
            std::ptr::write(hdr_ptr.add(1), cols as u64);

            let data = std::slice::from_raw_parts_mut(hdr_ptr.add(2) as *mut f64, nelems);

            for i in 0..nelems {
                data[i] = i as f64;
            }

            let matrix = Matrix::<f64>::from_buffer(ptr);

            assert_eq!(matrix.rows(), rows);
            assert_eq!(matrix.cols(), cols);
            assert_eq!(matrix.datalen(), nelems);

            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(matrix.get(i, j), (i * cols + j) as f64);
                }
            }


            dealloc(ptr as *mut u8, data_layout);

        }
    }


    #[test]
    fn test_matrix_to_buffer_seq()
    {
        let rows = 512;
        let cols = 1024;

        let nelems = rows * cols;

        let matrix = Matrix::from_fn(rows, cols, |i, j| (i * cols + j) as f64);
        let data_layout = Layout::array::<f64>(2 + nelems).expect("Layout creation failed");

        unsafe {

            // Calculate the total size: 2 u64s for rows and cols, then the f64 data
            let ptr = alloc(data_layout) as *mut c_void;

            if ptr.is_null() {
                panic!("Memory allocation failed!");
            }

            matrix.to_buffer_seq(ptr);

            let hdr_ptr = ptr as *const usize;
            assert_eq!(std::ptr::read(hdr_ptr), rows as usize);
            assert_eq!(std::ptr::read(hdr_ptr.add(1)), cols as usize);

            let data = std::slice::from_raw_parts(hdr_ptr.add(2) as *const f64, nelems);

            for i in 0..nelems {
                assert_eq!(data[i], i as f64);
            }

            dealloc(ptr as *mut u8, data_layout);
        }
    }

    #[test]
    fn test_matrix_to_buffer_par()
    {
        let rows = 512;
        let cols = 1024;

        let nelems = rows * cols;

        let matrix = Matrix::from_fn(rows, cols, |i, j| (i * cols + j) as f64);
        let data_layout = Layout::array::<f64>(2 + nelems).expect("Layout creation failed");

        unsafe {

            let ptr = alloc(data_layout) as *mut c_void;

            if ptr.is_null() {
                panic!("Memory allocation failed!");
            }

            matrix.to_buffer_parallel(ptr);

            let hdr_ptr = ptr as *mut u64;
            assert_eq!(std::ptr::read(hdr_ptr), rows as u64);
            assert_eq!(std::ptr::read(hdr_ptr.add(1)), cols as u64);

            let data = std::slice::from_raw_parts(hdr_ptr.add(2) as *const f64, nelems);

            for i in 0..nelems {
                assert_eq!(data[i], i as f64);
            }

            dealloc(ptr as *mut u8, data_layout);
        }
    }


    #[test]
    fn test_matrix_transpose_small_square_inplace()
    {
        let mut matrix = Matrix::from_fn(8, 8, |i, j| i * 8 + j);
        matrix.transpose_small_square_inplace();

        // Verify all elements
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(matrix.get(i, j), i + j * 8);
            }
        }
    }

    #[test]
    fn test_matrix_transpose_small_rectangle()
    {
        let matrix = Matrix::from_fn(16, 8, |i, j| i * 8 + j);
        let out = matrix.transpose_small_rectangle();

        // Verify all elements
        for i in 0..16 {
            for j in 0..8 {
                assert_eq!(out.get(j, i), matrix.get(i, j));
            }
        }
    }

    #[test]
    fn test_matrix_copy_to_block()
    {
        let matrix = Matrix::from_fn(64, 64, |i, j| ((i / 8) * 8 + (j / 8)));

        let mut block = Matrix::<usize>::new(8, 8);

        for i in 0..8 {
            for j in 0..8 {
                matrix.copy_to_block(&mut block, i * 8, j * 8);
                assert!(block.all(|&x| x == i * 8 + j));
            }
        }
    }

    #[test]
    fn test_matrix_from_to_block()
    {
        let mut matrix = Matrix::<usize>::new(64, 64);

        // Initialize the blocks and copy them to the main matrix
        for i in 0..8 {
            for j in 0..8 {
                let block = Matrix::from_fn(8, 8, |_, _| (i * 8 + j));
                matrix.copy_from_block(&block, i * 8, j * 8);
            }
        }

        assert!(matrix.validate());

        let mut block = Matrix::<usize>::new(8, 8);

        // Retrieve the blocks back and check (copy_to_block is
        // already tested, so I thrust it)
        for i in 0..8 {
            for j in 0..8 {
                matrix.copy_to_block(&mut block, i * 8, j * 8);
                assert!(block.all(|&x| x == i * 8 + j));
            }
        }
    }

    fn test_matrix_transpose<F>(test_fun: F, rows: usize, cols: usize)
    where
       F: for<'a> Fn(&'a Matrix<'a, usize>, usize) -> Matrix<'a, usize>
    {
        let matrix = Matrix::from_fn(rows, cols, |i, j| i * cols + j);
        assert!(matrix.validate());

        let transposed = test_fun(&matrix, 64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }


    #[test]
    fn test_matrix_transpose_big_squared()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize| mat.transpose_big(bsize),
            512, 512
        );
    }

    #[test]
    fn test_matrix_transpose_big_heigh()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize| mat.transpose_big(bsize),
            512, 128
        );
    }

    #[test]
    fn test_matrix_transpose_big_width()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize| mat.transpose_big(bsize),
            128, 512
        );
    }

    #[test]
    fn test_matrix_transpose_big_square_inplace()
    {
        let mut matrix = Matrix::from_fn(512, 512, |i, j| i * 512 + j);
        assert!(matrix.validate());

        matrix.transpose_parallel_square_inplace(64);
        assert!(matrix.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..512 {
                assert_eq!(matrix.get(i, j), j * 512 + i);
            }
        }
    }


    #[test]
    fn test_matrix_transpose_big_parallel_static_high()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize|  mat.transpose_parallel_static(bsize),
            512, 256
        );
    }

    #[test]
    fn test_matrix_transpose_big_parallel_static_width()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize|  mat.transpose_parallel_static(bsize),
            256, 512
        );
    }

    #[test]
    fn test_matrix_transpose_big_parallel_dynamic_high()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize|  mat.transpose_parallel_dynamic(bsize),
            512, 256
        );
    }

    #[test]
    fn test_matrix_transpose_big_parallel_dynamic_width()
    {
        test_matrix_transpose(
            |mat: &Matrix<usize>, bsize: usize|  mat.transpose_parallel_dynamic(bsize),
            256, 512
        );
    }
}
