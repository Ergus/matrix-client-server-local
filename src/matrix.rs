use std::ops::{Index, IndexMut};
use std::{ptr, fmt, cmp};
use rand::Rng;

use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::ffi::c_void;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Default + std::fmt::Debug, Standard: Distribution<T>
{
    const BLOCKDIM: usize = 64;  // This si a simple empiric value, we may tune it.

    /// Constructor
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![T::default(); rows * cols];
        Self { rows, cols, data }
    }

    /// Useful for the tests bellow
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    /// Useful for the tests bellow
    pub fn random(rows: usize, cols: usize) -> Self
    {
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..rows * cols).map(|_| rng.gen()).collect();
        Self { rows, cols,  data}
    }

    pub fn from_buffer(buffer: *const c_void) -> Self
    {
        let rows: usize = unsafe { *(buffer as *const usize) };
        let cols: usize = unsafe { *(buffer.add(8) as *const usize) };

        let mut data = vec![T::default(); rows * cols];

        unsafe {
            ptr::copy_nonoverlapping(
                buffer.add(16) as *const T,
                data.as_mut_ptr(),
                rows * cols
            );
        }

        Self {rows, cols,  data}
    }


    pub fn validate(&self) -> bool
    {
        self.rows > 16
            && self.cols >= 16
            && self.data.len() == self.rows * self.cols
    }

    /// Very primitive serialization function
    pub fn to_buffer(&self, buffer: *mut c_void)
    {
        unsafe {

            *(buffer as *mut usize) = self.rows;
            *(buffer.add(size_of::<usize>()) as *mut usize) = self.cols;

            ptr::copy_nonoverlapping(
                self.data.as_ptr(),
                buffer.add(2 * size_of::<usize>()) as *mut T,
                self.rows * self.cols
            );
        }
    }

    /// Get Rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// gcd is the min because we assume 2^n x 2^m matrices
    fn gcd(&self) -> usize {
        cmp::min(self.rows, self.cols)
    }


    /// Get Cols
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Get Ref
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Get Ref
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn payload_size(&self) -> usize
    {
        2 * size_of::<usize>() + self.rows * self.cols * size_of::<f64>()
    }


    fn copy_to_block(&self, block: &mut Matrix<T>, row_block: usize, col_block: usize)
    {
        assert_eq!(block.rows, block.cols, "block must be squared");

        let row_end: usize = (row_block + block.rows()).min(self.rows);

        let col_end: usize = (col_block + block.cols()).min(self.cols);
        let copysize: usize = col_end - col_block;

        let src: *const T = self.data.as_ptr();
        let dst: *mut T = block.data.as_mut_ptr();

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

    fn copy_from_block(&mut self, block: &Matrix<T>, row_block: usize, col_block: usize)
    {
        assert_eq!(block.rows, block.cols, "Block must be squared");
        assert!(block.rows <= self.gcd(), "Block dim must be <= gcd");
        assert!(self.gcd() % block.rows == 0  , "Block must be a divisor of gcd");

        let row_end: usize = (row_block + block.rows()).min(self.rows);

        let col_end: usize = (col_block + block.cols()).min(self.cols);
        let copysize: usize = col_end - col_block;

        let src: *mut T = self.data.as_mut_ptr();
        let dst: *const T = block.data.as_ptr();

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

    /// Full transpose in place for small matrices (intended to happen in the cache)
    fn transpose_small_square(&mut self)
    {
        assert_eq!(self.rows, self.cols, "Small transpose must be squared");

        for row in 0..self.rows {
            for col in 0..row {

                let tmp: T = self.data[col * self.rows + row].clone();
                self.data[col * self.rows + row] = self.data[row * self.cols + col].clone();
                self.data[row * self.cols + col] = tmp;
            }
        }
    }

    /// Full transpose in place for small matrices (intended to happen in the cache)
    fn transpose_small_rectangle(&self) -> Matrix<T>
    {
        assert!(self.rows <= 64, "Small rectangle tranpose rows must not exceed 64");
        assert!(self.cols <= 64, "Small rectangle tranpose rows must not exceed 64");

        let mut transpose = Matrix::<T>::new(self.cols, self.rows);

        for row in 0..self.rows {
            for col in 0..self.cols {
                transpose[(col, row)] = self[(row, col)].clone();
            }
        }

        transpose
    }


    // fn transpose_by_two_inplace(&mut self, row_block: usize, col_block: usize)
    // {
    //     let mut block1 = Matrix::<T>::new(blocksize, blocksize);
    //     let mut block2 = Matrix::<T>::new(blocksize, blocksize);

    //     self.copy_to_block(&mut block1, row_block, col_block);
    //     block1.transpose_small_square();

    //     self.copy_to_block(&mut block2, col_block, row_block);

    // }

    // Transpose
    pub fn transpose_big(&self, blocksize: usize) -> Matrix<T>
    {
        let mut transposed = Matrix::<T>::new(self.cols, self.rows);

        let mut block = Matrix::<T>::new(blocksize, blocksize);

        for row_block in (0..self.rows).step_by(blocksize) {
            for col_block in (0..self.cols).step_by(blocksize) {

                self.copy_to_block(&mut block, row_block, col_block);
                block.transpose_small_square();
                transposed.copy_from_block(&block, col_block, row_block);
            }
        }

        transposed
    }

    // Transpose
    pub fn transpose_parallel(&self, blocksize: usize) -> Matrix<T>
    {
        let mut transposed = Matrix::<T>::new(self.cols, self.rows);

        let mut block = Matrix::<T>::new(blocksize, blocksize);

        for row_block in (0..self.rows).step_by(blocksize) {
            for col_block in (0..self.cols).step_by(blocksize) {

                self.copy_to_block(&mut block, row_block, col_block);
                block.transpose_small_square();
                transposed.copy_from_block(&block, col_block, row_block);
            }
        }

        transposed
    }

    /// This is a simple heuristic, we may tune it 
    fn is_small_enough(&self) -> bool
    {
        self.data.len() < Self::BLOCKDIM * Self::BLOCKDIM
            || self.cols < Self::BLOCKDIM
            || self.rows < Self::BLOCKDIM
    }

    pub fn transpose(&self) -> Matrix<T>
    {
        if self.is_small_enough() {
            return self.transpose_small_rectangle()
        }

        self.transpose_big(Self::BLOCKDIM)
    }


}

/// Immutable indexing
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[row * self.cols + col]
    }
}

/// Mutable indexing
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[row * self.cols + col]
    }
}

/// Helper for print
impl<T: std::fmt::Debug> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            let slice = &self.data[i * self.cols.. (i + 1) * self.cols];

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
        let matrix = Matrix::<i32>::new(3, 4);

        // Verify dimensions
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);

        // Verify all elements
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(matrix[(i, j)], 0);
            }
        }
    }

    #[test]
    fn test_matrix_indexing()
    {
        let mut matrix = Matrix::<i32>::new(2, 2);

        // Set values
        matrix[(0, 0)] = 1;
        matrix[(0, 1)] = 2;
        matrix[(1, 0)] = 3;
        matrix[(1, 1)] = 4;

        // Verify values
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_getters()
    {
        let mut matrix = Matrix::<i32>::new(3, 3);

        // Use `get_mut` to modify a value
        if let Some(value) = matrix.get_mut(1, 1) {
            *value = 42;
        }

        // Verify with `get`
        assert_eq!(matrix.get(1, 1), Some(&42));
        assert_eq!(matrix.get(0, 0), Some(&0));
        assert_eq!(matrix.get(3, 3), None); // Out of bounds
    }

    #[test]
    fn test_matrix_clone_and_eq()
    {
        let matrix = Matrix::from_fn(2, 2, |i, j| i + j);
        let cloned_matrix = matrix.clone();

        // Verify equality
        assert_eq!(matrix, cloned_matrix);

        // Modify clone and verify inequality
        let mut modified_clone = cloned_matrix.clone();
        modified_clone[(0, 0)] = 99;
        assert_ne!(matrix, modified_clone);
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

        matrix.transpose_small_square();

        // Verify all elements
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(matrix[(i, j)], i + j * 8);
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
                assert_eq!(out[(j, i)], matrix[(i, j)]);
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
                assert!(block.data.iter().all(|&x| x == i * 8 + j));
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
                assert!(block.data.iter().all(|&x| x == i * 8 + j));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_squared()
    {
        let mut matrix = Matrix::from_fn(512, 512, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..512 {
                assert_eq!(transposed[(i, j)], matrix[(j, i)]);
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_random()
    {
        let mut matrix = Matrix::<f64>::random(512, 1024);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);

        // Verify all elements
        for i in 0..512 {
            for j in 0..1024 {
                assert_eq!(matrix[(i, j)], transposed[(j, i)]);
            }
        }
    }


    #[test]
    fn test_matrix_transpose_big_rectangular()
    {
        let mut matrix = Matrix::from_fn(512, 256, |i, j| i * 512 + j);
        assert!(matrix.validate());

        let transposed = matrix.transpose_big(64);
        assert!(transposed.validate());

        // Verify all elements
        for i in 0..512 {
            for j in 0..256 {
                assert_eq!(matrix[(i, j)], transposed[(j, i)]);
            }
        }
    }

}
