use crate::math::f32_cmp;

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};
use std::cmp::PartialEq;

#[derive(Clone, Copy, Debug)]
pub struct Tuple {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Tuple {
    // Constructors
    pub fn from(x: f32, y: f32, z: f32, w: f32) -> Self {
        Tuple {
            x: x,
            y: y,
            z: z,
            w: w,
        }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        // Here be bounds-checker abuse
        Tuple {
            x: slice[0],
            y: slice[1],
            z: slice[2],
            w: slice[3],
        }
    }

    pub fn new() -> Self {
        Self::from(0.0f32, 0.0f32, 0.0f32, 0.0f32)
    }

    pub fn make_point(x: f32, y: f32, z: f32) -> Self {
        Self::from(x, y, z, 1.0f32)
    }

    pub fn make_vector(x: f32, y: f32, z: f32) -> Self {
        Self::from(x, y, z, 0.0f32)
    }

    // Matrix Interop
    pub fn into_arr(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    // Comparisons
    pub fn is_point(&self) -> bool {
        f32_cmp(self.w, 1.0f32)
    }

    pub fn is_vector(&self) -> bool {
        f32_cmp(self.w, 0.0f32)
    }

    // Linear Algebra Functions
    pub fn mag(&self) -> f32 {
        f32::sqrt(
            self.x * self.x +
            self.y * self.y +
            self.z * self.z +
            self.w * self.w)
    }

    pub fn norm(&self) -> Self {
        let mut temp = self.clone();
        temp /= self.mag();
        temp
    }

    /*
    pub fn norm_assign(&mut self) {
        *self /= self.mag();
    }
    */

    pub fn dot(&self, rhs: Self) -> f32 {
        self.x * rhs.x +
        self.y * rhs.y +
        self.z * rhs.z +
        self.w * rhs.w
    }

    pub fn cross(&self, rhs: Self) -> Self {
        Self::make_vector(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x)
    }
}

impl Add for Tuple {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut temp = self.clone();
        temp += rhs;
        temp
    }
}

impl AddAssign for Tuple {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub for Tuple {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut temp = self.clone();
        temp -= rhs;
        temp
    }
}

impl SubAssign for Tuple {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Mul<f32> for Tuple {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        let mut temp = self.clone();
        temp *= rhs;
        temp
    }
}

impl MulAssign<f32> for Tuple {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl Div<f32> for Tuple {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        let mut temp = self.clone();
        temp /= rhs;
        temp
    }
}

impl DivAssign<f32> for Tuple {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

impl Neg for Tuple {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from(
            -self.x,
            -self.y,
            -self.z,
            -self.w)
    }
}

impl PartialEq for Tuple {
    fn eq(&self, rhs: &Self) -> bool {
        f32_cmp(self.x, rhs.x) &&
        f32_cmp(self.y, rhs.y) &&
        f32_cmp(self.z, rhs.z) &&
        f32_cmp(self.w, rhs.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ops() {
        let t1 = Tuple::from(3f32, -2f32, 5f32, 1f32);
        let t2 = Tuple::from(-2f32, 3f32, 1f32, 0f32);
        let mut t3 = t1.clone();
        let exp = Tuple::from(1f32, 1f32, 6f32, 1f32);
        
        assert_eq!(t1 + t2, exp);
        
        t3 += t2;
        assert_eq!(t3, exp);
    }

    #[test]
    fn sub_ops() {
        let p1 = Tuple::make_point(3f32, 2f32, 1f32);
        let p2 = Tuple::make_point(5f32, 6f32, 7f32);
        let mut p3 = p1.clone();
        let exp = Tuple::make_vector(-2f32, -4f32, -6f32);

        assert_eq!(p1 - p2, exp);

        p3 -= p2;
        assert_eq!(p3, exp);
    }

    #[test]
    fn mul_ops() {
        let t = Tuple::from(1f32, -2f32, 3f32, -4f32);
        let exp1 = Tuple::from(3.5f32, -7f32, 10.5f32, -14f32);
        let exp2 = Tuple::from(0.5f32, -1f32, 1.5f32, -2f32);

        assert_eq!(t * 3.5f32, exp1);
        assert_eq!(t * 0.5f32, exp2);

        let mut t1 = t.clone();
        t1 *= 3.5f32;
        assert_eq!(t1, exp1);

        let mut t2 = t.clone();
        t2 *= 0.5f32;
        assert_eq!(t2, exp2);
    }

    #[test]
    fn div_ops() {
        let t1 = Tuple::from(1f32, -2f32, 3f32, -4f32);
        let mut t2 = Tuple::from(1f32, -2f32, 3f32, -4f32);
        let exp = Tuple::from(0.5f32, -1f32, 1.5f32, -2f32);

        assert_eq!(t1 / 2f32, exp);

        t2 /= 2f32;
        assert_eq!(t2, exp);
    }

    #[test]
    fn neg_op() {
        let t = Tuple::from(1f32, -2f32, 3f32, -4f32);
        let exp = Tuple::from(-1f32, 2f32, -3f32, 4f32);

        assert_eq!(-t, exp);
    }

    #[test]
    fn linear_ops() {
        let m_v1 = Tuple::make_vector(1f32, 0f32, 0f32);
        let m_v2 = Tuple::make_vector(0f32, 1f32, 0f32);
        let m_v3 = Tuple::make_vector(0f32, 0f32, 1f32);
        let m_v4 = Tuple::make_vector(1f32, 2f32, 3f32);
        let m_v5 = Tuple::make_vector(-1f32, -2f32, -3f32);

        let m_exp123 = 1f32;
        let m_exp45 = f32::sqrt(14f32);

        assert_eq!(m_v1.mag(), m_exp123);
        assert_eq!(m_v2.mag(), m_exp123);
        assert_eq!(m_v3.mag(), m_exp123);

        assert_eq!(m_v4.mag(), m_exp45);
        assert_eq!(m_v5.mag(), m_exp45);


        let n_v1 = Tuple::make_vector(4f32, 0f32, 0f32);
        let n_v2 = Tuple::make_vector(1f32, 2f32, 3f32);

        let n_exp1 = Tuple::make_vector(1f32, 0f32, 0f32);
        let n_exp2 = Tuple::make_vector(
            1f32 / f32::sqrt(14f32),
            2f32 / f32::sqrt(14f32),
            3f32 / f32::sqrt(14f32));

        assert_eq!(n_v1.norm(), n_exp1);
        assert_eq!(n_v2.norm(), n_exp2);

        assert!(f32_cmp(n_v1.norm().mag(), 1f32));


        let d_v1 = Tuple::make_vector(1f32, 2f32, 3f32);
        let d_v2 = Tuple::make_vector(2f32, 3f32, 4f32);

        let d_exp = 20f32;

        assert_eq!(d_v1.dot(d_v2), d_exp); 
        assert_eq!(d_v2.dot(d_v1), d_exp);


        let c_v1 = Tuple::make_vector(1f32, 2f32, 3f32);
        let c_v2 = Tuple::make_vector(2f32, 3f32, 4f32);

        let c_exp1 = Tuple::make_vector(-1f32, 2f32, -1f32);
        let c_exp2 = -c_exp1;

        assert_eq!(c_v1.cross(c_v2), c_exp1);
        assert_eq!(c_v2.cross(c_v1), c_exp2);
    }

    #[test]
    fn constructors() {
        let z = Tuple::new();
        let z_exp = Tuple::from(0f32, 0f32, 0f32, 0f32);

        assert_eq!(z, z_exp);


        let p = Tuple::make_point(4f32, -4f32, 3f32);
        let p_exp = Tuple::from(4f32, -4f32, 3f32, 1f32);

        assert_eq!(p, p_exp);
        assert!(p.is_point());


        let v = Tuple::make_vector(4f32, -4f32, 3f32);
        let v_exp = Tuple::from(4f32, -4f32, 3f32, 0f32);

        assert_eq!(v, v_exp);
        assert!(v.is_vector());
    }
}
