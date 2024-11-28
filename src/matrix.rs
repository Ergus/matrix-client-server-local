use std::ops::{Index, IndexMut};
use std::{ptr, fmt, cmp};
use rand::Rng;

use rand::distributions::Standard;
use rand::prelude::Distribution;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Default, Standard: Distribution<T>
{
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


    fn copy_to_buffer(&self, buffer: &mut Matrix<T>, row_block: usize, col_block: usize)
    {
        assert_eq!(buffer.rows, buffer.cols, "buffer must be squared");

        let row_end: usize = (row_block + buffer.rows()).min(self.rows);

        let col_end: usize = (col_block + buffer.cols()).min(self.cols);
        let copysize: usize = col_end - col_block;

        let src: *const T = self.data.as_ptr();
        let dst: *mut T = buffer.data.as_mut_ptr();

        let mut startdst: usize = 0;

        // Copy from matrix to buffers
        for row in row_block..row_end {

            unsafe {
                // Efficient vectorized copy (~memcpy)
                ptr::copy_nonoverlapping(
                    src.add(row * self.cols + col_block),
                    dst.add(startdst),
                    copysize);
            }

            startdst += buffer.cols();
        }
    }

    fn copy_from_buffer(&mut self, buffer: &Matrix<T>, row_block: usize, col_block: usize)
    {
        assert_eq!(buffer.rows, buffer.cols, "Buffer must be squared");
        assert!(buffer.rows <= self.gcd(), "Buffer dim must be <= gcd");
        assert!(self.gcd() % buffer.rows == 0  , "Buffer must be a divisor of gcd");

        let row_end: usize = (row_block + buffer.rows()).min(self.rows);

        let col_end: usize = (col_block + buffer.cols()).min(self.cols);
        let copysize: usize = col_end - col_block;

        let src: *mut T = self.data.as_mut_ptr();
        let dst: *const T = buffer.data.as_ptr();

        let mut startdst: usize = 0;

        // Copy from matrix to buffers
        for row in row_block..row_end {

            unsafe {
                // Efficient vectorized copy (~memcpy)
                ptr::copy_nonoverlapping(
                    dst.add(startdst),
                    src.add(row * self.cols + col_block),
                    copysize);
            }

            startdst += buffer.cols();
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


    // fn transpose_by_two_inplace(&mut self, row_block: usize, col_block: usize)
    // {
    //     let mut buffer1 = Matrix::<T>::new(blocksize, blocksize);
    //     let mut buffer2 = Matrix::<T>::new(blocksize, blocksize);

    //     self.copy_to_buffer(&mut buffer1, row_block, col_block);
    //     buffer1.transpose_small_square();

    //     self.copy_to_buffer(&mut buffer2, col_block, row_block);

    // }

    // Transpose
    pub fn transpose(&mut self, blocksize: usize) -> Matrix<T>
    {
        let mut transposed = Matrix::<T>::new(self.cols, self.rows);

        let mut buffer = Matrix::<T>::new(blocksize, blocksize);

        for row_block in (0..self.rows).step_by(blocksize) {
            for col_block in (0..self.cols).step_by(blocksize) {

                self.copy_to_buffer(&mut buffer, row_block, col_block);
                buffer.transpose_small_square();
                transposed.copy_from_buffer(&buffer, col_block, row_block);
            }
        }

        transposed
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
            let slice = &self.data[i * self.rows.. (i + 1) * self.rows];

            write!(f, "{:?}\n", slice)?;  // Format each row and move to the next line
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_matrix_constructor() {
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
    fn test_matrix_indexing() {
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
    fn test_matrix_getters() {
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
    fn test_matrix_clone_and_eq() {
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
    fn test_matrix_debug() {
        let matrix = Matrix::from_fn(2, 2, |i, j| i * j);

        // Ensure debug string is as expected
        let debug_output = format!("{:?}", matrix);
        assert!(debug_output.contains("Matrix"));
        assert!(debug_output.contains("data:"));
    }


    #[test]
    fn test_matrix_transpose_small() {
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
    fn test_matrix_copy_to_buffer() {
        let matrix = Matrix::from_fn(64, 64, |i, j| ((i / 8) * 8 + (j / 8)));

        let mut buffer = Matrix::<usize>::new(8, 8);


        for i in 0..8 {
            for j in 0..8 {
                matrix.copy_to_buffer(&mut buffer, i * 8, j * 8);
                assert!(buffer.data.iter().all(|&x| x == i * 8 + j));
            }
        }
    }


    #[test]
    fn test_matrix_from_buffer() {

        let mut matrix = Matrix::<usize>::new(64, 64);

        //let matrix = Matrix::from_fn(64, 64, |i, j| ((i / 8) * 8 + (j / 8)));


        for i in 0..8 {
            for j in 0..8 {
                let buffer = Matrix::from_fn(8, 8, |_, _| (i * 8 + j));
                matrix.copy_from_buffer(&buffer, i * 8, j * 8);
            }
        }


        let mut buffer = Matrix::<usize>::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                matrix.copy_to_buffer(&mut buffer, i * 8, j * 8);
                assert!(buffer.data.iter().all(|&x| x == i * 8 + j));
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_squared() {
        let mut matrix = Matrix::from_fn(512, 512, |i, j| i * 512 + j);

        let transposed = matrix.transpose(64);

        // Verify all elements
        for i in 0..512 {
            for j in 0..512 {
                assert_eq!(transposed[(i, j)], matrix[(j, i)]);
            }
        }
    }

    #[test]
    fn test_matrix_transpose_big_random() {
        let mut matrix = Matrix::<f64>::random(512, 1024);

        let transposed = matrix.transpose(64);

        // Verify all elements
        for i in 0..512 {
            for j in 0..1024 {
                assert_eq!(matrix[(i, j)], transposed[(j, i)]);
            }
        }
    }


    #[test]
    fn test_matrix_transpose_big_rectangular() {
        let mut matrix = Matrix::from_fn(512, 256, |i, j| i * 512 + j);

        let transposed = matrix.transpose(64);

        // Verify all elements
        for i in 0..512 {
            for j in 0..256 {
                assert_eq!(matrix[(i, j)], transposed[(j, i)]);
            }
        }
    }

}
