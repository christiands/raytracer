use crate::math::f32_cmp;
use crate::tuple::Tuple;

use std::cmp::PartialEq;
use std::ops::{Mul, MulAssign};

// TODO: Figure out why rustc flags this as dead code even though it obviously isn't
macro_rules! matrix_gen {
    ($name: ident, $rows: literal, $cols: literal) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            data: [f32; $rows * $cols],
        }

        impl $name {
            fn new() -> Self {
                Self {
                    data: [0f32; $rows * $cols],
                }
            }

            fn from_slice(slice: &[f32]) -> Self {
                if slice.len() < ($rows * $cols) {
                    panic!("Slice is not long enough to utilize: got {}, expected {}!",
                                slice.len(), $rows * $cols);
                }
                else {
                    let mut temp = [0f32; $rows * $cols];
                    temp.copy_from_slice(&slice[..$rows * $cols]);

                    Self {
                        data: temp,
                    }
                }
            }

            // Slicing functions for easier math
            fn col(&self, col: usize) -> [f32; $rows] {
                if col >= $cols {
                    panic!("Selected column exceeds matrix size! Expected <{}, got {}!",
                           $cols, col);
                }
                else {
                    let mut temp = [0f32; $rows];

                    for i in 0..$rows {
                        temp[i] = self.data[col + i * $rows];
                    }

                    temp
                }
            }

            fn row(&self, row: usize) -> [f32; $cols] {
                if row >= $rows {
                    panic!("Selected column exceeds matrix size! Expected <{}, got {}!",
                           $rows, row);
                }
                else {
                    let mut temp = [0f32; $rows];
                    temp.copy_from_slice(&self.data[(row * $cols)..(row * $cols + $cols)]);

                    temp
                }
            }
        }

        impl Matrix for $name {
            fn at(&mut self, row: usize, col: usize) -> &mut f32 {
                if row >= $rows || col >= $cols {
                    panic!("Invalid coordinate ({}, {}) for a {}x{} matrix!", row, col, $rows, $cols);
                }
                &mut self.data[row * $cols + col]
            }
        }

        impl PartialEq for $name {
            fn eq(&self, rhs: &Self) -> bool {
                self.data.iter().zip(rhs.data.iter()).all(|(a, b)| f32_cmp(*a, *b))
            }
        }
    };
}

pub trait Matrix {
    // This function is designed to panic if OOB (it just uses Rust's intrinsic OOB checker)
    // Efficient utilization is just a reference to the array element
    fn at(&mut self, row: usize, col: usize) -> &mut f32;
}

matrix_gen!(Mat2D, 2, 2);

impl Mat2D {
    fn determinant(&self) -> f32 {
        self.data[0] * self.data[3] - self.data[1] * self.data[2]
    }
}

matrix_gen!(Mat3D, 3, 3);

impl Mat3D {
    fn submatrix(&self, row: usize, col: usize) -> Mat2D {
        let mut temp = Mat2D::new();
        let mut temp_col = 0;
        let mut temp_row = 0;

        for r in 0..3 {
            if r != row {
                for c in 0..3 {
                    if c != col {
                        temp.data[temp_row * 2 + temp_col] = self.data[r * 3 + c];
                        temp_col += 1;
                    }
                }
                temp_row += 1;
                temp_col = 0;
            }
        }

        temp
    }

    fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 == 1 {
            -self.minor(row, col)
        }
        else {
            self.minor(row, col)
        }
    }

    fn determinant(&self) -> f32 {
        let mut temp = 0f32;

        (0..3).for_each(|i| { temp += self.data[i] * self.cofactor(0, i); });

        temp
    }
}

matrix_gen!(Mat4D, 4, 4);

impl Mat4D {
    // The amount of hand-coded optimizing going on is a little nerve-racking
    // I really don't trust compilers too much
    fn identity() -> Self {
        Self {
            data: [1f32, 0f32, 0f32, 0f32,
                   0f32, 1f32, 0f32, 0f32,
                   0f32, 0f32, 1f32, 0f32,
                   0f32, 0f32, 0f32, 1f32],
        }
    }

    fn transpose(&self) -> Self {
        let mut temp = Self::new();

        // I'm thinking of making a competition for the longest Rust function chain
        for col in 0..4 {
            let offset = col * 4;
            temp.data[offset..(offset + 4)].copy_from_slice(&self.col(col));
        }

        temp
    }

    fn submatrix(&self, row: usize, col: usize) -> Mat3D {
        let mut temp = Mat3D::new();
        let mut temp_col = 0;
        let mut temp_row = 0;

        for r in 0..4 {
            if r != row {
                for c in 0..4 {
                    if c != col {
                        temp.data[temp_row * 3 + temp_col] = self.data[r * 4 + c];
                        temp_col += 1;
                    }
                }
                temp_row += 1;
                temp_col = 0;
            }
        }

        temp
    }

    fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 == 1 {
            -self.minor(row, col)
        }
        else {
            self.minor(row, col)
        }
    }

    fn determinant(&self) -> f32 {
        let mut temp = 0f32;

        (0..4).for_each(|i| { temp += self.data[i] * self.cofactor(0, i); });

        temp
    }

    fn col_tuple(&self, col: usize) -> Tuple {
        Tuple::from_slice(&self.col(col))
    }

    fn row_tuple(&self, row: usize) -> Tuple {
        Tuple::from_slice(&self.row(row))
    }

    fn is_invertible(&self) -> bool {
        !f32_cmp(self.determinant(), 0f32)
    }

    fn invert(&self) -> Self {
        if !self.is_invertible() {
            panic!("Matrix \"{:?}\" is not invertible!", self);
        }

        let mut temp = Self::new();

        for row in 0..4 {
            for col in 0..4 {
                // This is a column-major index, rather than a row-major one
                // The only reason is because swapping majors results in automatic transposing
                let index = col * 4 + row;
                temp.data[index] = self.cofactor(row, col) / self.determinant();
            }
        }

        temp
    }
}

impl Mul for Mat4D {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let dot_arrays = |a: &[f32], b: &[f32]| -> f32 {
            let mut result = 0f32;

            a.iter().zip(b.iter()).for_each(|(c, d)| { result += c * d; } );

            result
        };

        let mut temp = Self::new();

        // Very not designed for efficiency (row() is relatively fast, col() is slow as balls)
        for row in 0..4 {
            for col in 0..4 {
                let index = row * 4 + col;

                temp.data[index] = dot_arrays(&self.row(row), &rhs.col(col));
            }
        }

        temp
    }
}

impl Mul<Tuple> for Mat4D {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Tuple {
        let mut temp = [0f32; 4];

        for row in 0..4 {
            // This is actually some nasty looking work lmao
            // If the compiler is stupid (which it shouldn't be), it'll call into_arr() four times
            //   which would be stupid (rhs never changes, and the conversion between Tuple and
            //   array isn't exactly free)
            self.row(row).iter().zip(rhs.into_arr().iter())
                .for_each(|(a, b)| { temp[row] += a * b; });
        }

        Tuple::from_slice(&temp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison() {
        let m1 = Mat4D::from_slice(&[1.0, 2.0, 3.0, 4.0,
                                     5.0, 6.0, 7.0, 8.0,
                                     9.0, 8.0, 7.0, 6.0,
                                     5.0, 4.0, 3.0, 2.0]);
        let m2 = m1.clone();
        let m3 = Mat4D::from_slice(&[2.0, 3.0, 4.0, 5.0,
                                     6.0, 7.0, 8.0, 9.0,
                                     8.0, 7.0, 6.0, 5.0,
                                     4.0, 3.0, 2.0, 1.0]);

        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }
    
    #[test]
    fn mat2d_determinant() {
        let m = Mat2D::from_slice(&[1.0, 5.0,
                                   -3.0, 2.0]);

        assert!(f32_cmp(m.determinant(), 17.0));
    }

    #[test]
    fn mat3d_determinant() {
        let m = Mat3D::from_slice(&[1.0, 2.0,  6.0,
                                   -5.0, 8.0, -4.0,
                                    2.0, 6.0,  4.0]);

        assert!(f32_cmp(m.determinant(), -196.0));
    }

    #[test]
    fn mat3d_minor() {
        let m = Mat3D::from_slice(&[3.0,  5.0,  0.0,
                                    2.0, -1.0, -7.0,
                                    6.0, -1.0,  5.0]);

        assert_eq!(m.minor(1, 0), 25.0);
        assert!(f32_cmp(m.submatrix(1, 0).determinant(), m.minor(1, 0)));
    }

    #[test]
    fn mat3d_submatrix() {
        let m = Mat3D::from_slice(&[ 1.0, 5.0,  0.0,
                                    -3.0, 2.0,  7.0,
                                     0.0, 6.0, -3.0]);

        let exp = Mat2D::from_slice(&[-3.0, 2.0,
                                       0.0, 6.0]);

        assert_eq!(m.submatrix(0, 2), exp);
    }

    // TODO: add a test for the cofactors of this matrix
    #[test]
    fn mat4d_determinant() {
        let m = Mat4D::from_slice(&[-2.0, -8.0,  3.0,  5.0,
                                    -3.0,  1.0,  7.0,  3.0,
                                     1.0,  2.0, -9.0,  6.0,
                                    -6.0,  7.0,  7.0, -9.0]);

        assert!(f32_cmp(m.determinant(), -4071.0));
    }

    #[test]
    fn mat4d_submatrix() {
        let m = Mat4D::from_slice(&[-6.0, 1.0,  1.0, 6.0,
                                    -8.0, 5.0,  8.0, 6.0,
                                    -1.0, 0.0,  8.0, 2.0,
                                    -7.0, 1.0, -1.0, 1.0]);

        let exp = Mat3D::from_slice(&[-6.0,  1.0, 6.0,
                                      -8.0,  8.0, 6.0,
                                      -7.0, -1.0, 1.0]);

        assert_eq!(m.submatrix(2, 1), exp);
    }

    #[test]
    fn mat4d_mulmat4d() {
        let m1 = Mat4D::from_slice(&[1.0, 2.0, 3.0, 4.0,
                                     5.0, 6.0, 7.0, 8.0,
                                     9.0, 8.0, 7.0, 6.0,
                                     5.0, 4.0, 3.0, 2.0]);
        let m2 = Mat4D::from_slice(&[-2.0, 1.0, 2.0,  3.0,
                                      3.0, 2.0, 1.0, -1.0,
                                      4.0, 3.0, 6.0,  5.0,
                                      1.0, 2.0, 7.0,  8.0]);
        
        let exp = Mat4D::from_slice(&[20.0, 22.0,  50.0,  48.0,
                                      44.0, 54.0, 114.0, 108.0,
                                      40.0, 58.0, 110.0, 102.0,
                                      16.0, 26.0,  46.0,  42.0]);

        assert_eq!(m1 * m2, exp);
    }

    #[test]
    fn mat4d_multuple() {
        let m = Mat4D::from_slice(&[1.0, 2.0, 3.0, 4.0,
                                    2.0, 4.0, 4.0, 2.0,
                                    8.0, 6.0, 4.0, 1.0,
                                    0.0, 0.0, 0.0, 1.0]);
        let t = Tuple::from(1.0, 2.0, 3.0, 1.0);

        let exp = Tuple::from(18.0, 24.0, 33.0, 1.0);

        assert_eq!(m * t, exp);
    }

    #[test]
    fn mat4d_identity() {
        let m1 = Mat4D::from_slice(&[0.0, 1.0,  2.0,  4.0,
                                     1.0, 2.0,  4.0,  8.0,
                                     2.0, 4.0,  8.0, 16.0,
                                     4.0, 8.0, 16.0, 32.0]);
        let m2 = Mat4D::identity();

        let exp = m1.clone();

        assert_eq!(m1 * m2, exp);
    }

    #[test]
    fn mat4d_transpose() {
        let m = Mat4D::from_slice(&[0.0, 9.0, 3.0, 0.0,
                                    9.0, 8.0, 0.0, 8.0,
                                    1.0, 8.0, 5.0, 3.0,
                                    0.0, 0.0, 5.0, 8.0]);

        let exp = Mat4D::from_slice(&[0.0, 9.0, 1.0, 0.0,
                                      9.0, 8.0, 8.0, 0.0,
                                      3.0, 0.0, 5.0, 5.0,
                                      0.0, 8.0, 3.0, 8.0]);

        assert_eq!(m.transpose(), exp);
    }

    #[test]
    fn mat4d_invertible() {
        let m1 = Mat4D::from_slice(&[6.0,  4.0, 4.0,  4.0,
                                     5.0,  5.0, 7.0,  6.0,
                                     4.0, -9.0, 3.0, -7.0,
                                     9.0,  1.0, 7.0, -6.0]);
        let m2 = Mat4D::from_slice(&[-4.0,  2.0, -2.0, -3.0,
                                      9.0,  6.0,  2.0,  6.0,
                                      0.0, -5.0,  1.0, -5.0,
                                      0.0,  0.0,  0.0,  0.0]);

        assert!(m1.is_invertible());
        assert!(!m2.is_invertible());
    }

    #[test]
    fn mat4d_inversion() {
        // The test-case matrixes are from pages 39 and 41
        let m1 = Mat4D::from_slice(&[-5.0,  2.0,  6.0, -8.0,
                                      1.0, -5.0,  1.0,  8.0,
                                      7.0,  7.0, -6.0, -7.0,
                                      1.0, -3.0,  7.0,  4.0]);
        let m2 = Mat4D::from_slice(&[ 8.0, -5.0,  9.0,  2.0,
                                      7.0,  5.0,  6.0,  1.0,
                                     -6.0,  0.0,  9.0,  6.0,
                                     -3.0,  0.0, -9.0, -4.0]);
        let m3 = Mat4D::from_slice(&[ 9.0,  3.0,  0.0,  9.0,
                                     -5.0, -2.0, -6.0, -3.0,
                                     -4.0,  9.0,  6.0,  4.0,
                                     -7.0,  6.0,  6.0,  2.0]);

        let exp1 = Mat4D::from_slice(&[ 0.21805,  0.45113,  0.24060, -0.04511,
                                       -0.80827, -1.45677, -0.44361,  0.52068,
                                       -0.07895, -0.22368, -0.05263,  0.19737,
                                       -0.52256, -0.81391, -0.30075,  0.30639]);
        let exp2 = Mat4D::from_slice(&[-0.15385, -0.15385, -0.28205, -0.53846,
                                       -0.07692,  0.12308,  0.02564,  0.03077,
                                        0.35897,  0.35897,  0.43590,  0.92308,
                                       -0.69231, -0.69231, -0.76923, -1.92308]);
        let exp3 = Mat4D::from_slice(&[-0.04074, -0.07778,  0.14444, -0.22222,
                                       -0.07778,  0.03333,  0.36667, -0.33333,
                                       -0.02901, -0.14630, -0.10926,  0.12963,
                                        0.17778,  0.06667, -0.26667,  0.33333]);

        assert_eq!(m1.invert(), exp1);
        assert_eq!(m2.invert(), exp2);
        assert_eq!(m3.invert(), exp3);
    }
}
