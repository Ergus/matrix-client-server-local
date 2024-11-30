use std::{ptr, fmt, cmp};
use std::sync::{Arc, RwLock};
use std::ffi::c_void;
use std::fmt::Debug;

use rand::Rng;
use rand::distributions::Standard;
use rand::prelude::Distribution;

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
    Debug + 
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

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    /// Number of rows
    rows: usize,

    /// Number of columns
    cols: usize,

    /// The matrix data stored in an Arc
    data: Arc<RwLock<Vec<T>>>,
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
impl<T> Matrix<T>
where
    T: Numeric64, Standard: Distribution<T>
{
    const BLOCKDIM: usize = 64;  // This si a simple empiric value, we may tune it.

    /// Basic constructor to create an empty matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = Arc::new(RwLock::new(vec![T::default(); rows * cols]));

        Self { rows, cols, data }
    }

    /// Constructor to generate the matrix based on an iteration function.
    /// This is specially useful for the tests
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let data = Arc::new(RwLock::new(Vec::with_capacity(rows * cols)));

        {
            let mut wguard = data.write().unwrap();

            for i in 0..rows {
                for j in 0..cols {
                    wguard.push(f(i, j));
                }
            }
        }
        Self { rows, cols, data }
    }

    /// Function to generate the random matrices.
    /// This is the function used in the client
    pub fn random(rows: usize, cols: usize) -> Self
    {
        let mut rng = rand::thread_rng();

        let data = Arc::new(RwLock::new((0..rows * cols).map(|_| rng.gen()).collect()));

        Self { rows, cols, data}
    }

    /// Copy the matrix from a memory buffer. Generally used to copy
    /// from shared memory
    pub fn from_buffer(buffer: *const c_void) -> Self
    {
        let rows: usize = unsafe { *(buffer as *const usize) };
        let cols: usize = unsafe { *(buffer.add(8) as *const usize) };

        let data = Arc::new(RwLock::new(vec![T::default(); rows * cols]));

        unsafe {
            let mut wguard = data.write().unwrap();

            ptr::copy_nonoverlapping(
                buffer.add(16) as *const T,
                wguard.as_mut_ptr(),
                rows * cols
            );

        }
        Self {rows, cols,  data}
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
    pub fn to_buffer(&self, buffer: *mut c_void)
    {
        unsafe {
            let rguard = self.data.read().unwrap();

            *(buffer as *mut usize) = self.rows;
            *(buffer.add(size_of::<usize>()) as *mut usize) = self.cols;

            ptr::copy_nonoverlapping(
                rguard.as_ptr(),
                buffer.add(2 * size_of::<usize>()) as *mut T,
                self.rows * self.cols
            );
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

        for row in 0..self.rows {
            for col in 0..row {

                let tmp: T = wguard[col * self.rows + row].clone();
                wguard[col * self.rows + row] = wguard[row * self.cols + col].clone();
                wguard[row * self.cols + col] = tmp;
            }
        }
    }

    /// Full transpose for small matrices without blocks.
    pub fn transpose_small_rectangle(&self) -> Matrix<T>
    {
        assert!(self.rows <= 64, "Small rectangle tranpose rows must not exceed 64");
        assert!(self.cols <= 64, "Small rectangle tranpose rows must not exceed 64");

        let transpose = Matrix::<T>::new(self.cols, self.rows);

        {
            let rguard = self.data.read().unwrap();
            let mut twguard = transpose.data.write().unwrap();

            for row in 0..self.rows {
                for col in 0..self.cols {
                    twguard[col * self.rows + row] = rguard[row * self.cols + col].clone();
                }
            }
        }

        transpose
    }

    /// Full transpose for big matrices with blocks, but without threads.
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
    pub fn transpose_parallel(&self, blocksize: usize) -> Matrix<T>
    {
        // TODO: Improve this
        let n_threads = 8;
        let transposed = Matrix::<T>::new(self.cols, self.rows);

        let blocks_cols = self.cols / blocksize;
        let total_blocks = (self.rows / blocksize) * blocks_cols;

        let blocks_per_thread = total_blocks / n_threads;
        let blocks_rest = total_blocks % n_threads;  // This is likely to be zero

        std::thread::scope(|s| {

            for i in 0..n_threads {

                let cself = self.clone();
                let mut ctran = transposed.clone();

                let _ = s.spawn(move || {

                    let mut block = Matrix::<T>::new(blocksize, blocksize);

                    let first_block_thread = i * blocks_per_thread + cmp::min(i, blocks_rest);
                    let nblocks_thread = blocks_per_thread + ((i < blocks_rest) as usize);

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

    /// This is a simple heuristic to decide when to use the
    /// un-blocked version, we may tune it.
    fn is_small_enough(&self) -> bool
    {
        self.cols * self.rows < Self::BLOCKDIM * Self::BLOCKDIM
            || self.cols < Self::BLOCKDIM
            || self.rows < Self::BLOCKDIM
    }

    /// This is intended to become the main function to use in the
    /// server code.
    ///
    /// # TODO
    /// Still needs improve the heuristics
    pub fn transpose(&self) -> Matrix<T>
    {
        if self.is_small_enough() {
            return self.transpose_small_rectangle()
        }

        self.transpose_big(Self::BLOCKDIM)
    }

    /// Get a matrix value using copy.
    pub fn get(&self, row: usize, col: usize) -> T
    {
        let rguard = self.data.read().unwrap();
        rguard[row * self.cols + col]
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

        Matrix::<T>::from_fn(
            self.rows,
            self.cols,
            |i, j| rguard1[i * self.cols + j] - rguard2[i * self.cols + j]
        )
    }
}

/// Operator ==
impl<T: std::cmp::PartialEq> PartialEq<Matrix<T>> for Matrix<T> {

    fn eq(&self, other: &Matrix<T>) -> bool {
        let rguard1 = self.data.read().unwrap();
        let rguard2 = other.data.read().unwrap();

        return self.rows == other.rows
            && self.cols == other.cols
            && *rguard1 == *rguard2;
    }
}


/// Helper for print
impl<T: std::fmt::Debug> fmt::Display for Matrix<T> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let rguard = self.data.read().unwrap();

        for i in 0..self.rows {
            let slice = &rguard[i * self.cols.. (i + 1) * self.cols];

            write!(f, "{:?}\n", slice)?;  // Format each row and move to the next line
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_matrix_constructor()
    {
        let matrix = Matrix::<i64>::new(3, 4);

        // Verify dimensions
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);

        // Verify all elements
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

        // Ensure debug string is as expected
        let debug_output = format!("{:?}", matrix);
        assert!(debug_output.contains("Matrix"));
        assert!(debug_output.contains("data:"));
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
    fn test_matrix_from_block()
    {

        let mut matrix = Matrix::<usize>::new(64, 64);

        for i in 0..8 {
            for j in 0..8 {
                let block = Matrix::from_fn(8, 8, |_, _| (i * 8 + j));
                matrix.copy_from_block(&block, i * 8, j * 8);
            }
        }

        assert!(matrix.validate());

        let mut block = Matrix::<usize>::new(8, 8);

        for i in 0..8 {
            for j in 0..8 {
                matrix.copy_to_block(&mut block, i * 8, j * 8);
                assert!(block.all(|&x| x == i * 8 + j));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_squared()
    {
        let matrix = Matrix::from_fn(512, 512, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..512 {
                assert_eq!(transposed.get(i, j), matrix.get(j, i));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_random()
    {
        let matrix = Matrix::<f64>::random(512, 1024);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);

        // Verify all elements
        for i in 0..512 {
            for j in 0..1024 {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_rectangular()
    {
        let matrix = Matrix::from_fn(512, 256, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..256 {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }


    #[test]
    fn test_matrix_transpose_big_parallel_high()
    {
        let matrix = Matrix::from_fn(512, 256, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_parallel(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..256 {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_parallel_width()
    {
        let matrix = Matrix::from_fn(256, 512, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_parallel(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..256 {
            for j in 0..512 {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }

}
