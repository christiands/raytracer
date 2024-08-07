use crate::math::f32_cmp;

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign};
use std::cmp::PartialEq;

#[derive(Clone, Copy, Debug)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    pub fn from(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r,
            g: g,
            b: b,
        }
    }

    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::from(
            r as f32 / u8::MAX as f32,
            g as f32 / u8::MAX as f32,
            b as f32 / u8::MAX as f32)
    }

    pub fn new() -> Self {
        Self::from(0f32, 0f32, 0f32)
    }

    pub fn as_rgb8_bytes(&self) -> [u8; 3] {
        [(self.r * 255f32) as u8, (self.g * 255f32) as u8, (self.b * 255f32) as u8]
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut temp = self.clone();
        temp += rhs;
        temp
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, rhs: Self) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut temp = self.clone();
        temp -= rhs;
        temp
    }
}

impl SubAssign for Color {
    fn sub_assign(&mut self, rhs: Self) {
        self.r -= rhs.r;
        self.g -= rhs.g;
        self.b -= rhs.b;
    }
}

impl Mul<f32> for Color {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        let mut temp = self.clone();
        temp *= rhs;
        temp
    }
}

impl MulAssign<f32> for Color {
    fn mul_assign(&mut self, rhs: f32) {
        self.r *= rhs;
        self.g *= rhs;
        self.b *= rhs;
    }
}

impl Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut temp = self.clone();
        temp *= rhs;
        temp
    }
}

impl MulAssign for Color {
    fn mul_assign(&mut self, rhs: Self) {
        self.r *= rhs.r;
        self.g *= rhs.g;
        self.b *= rhs.b;
    }
}

impl PartialEq for Color {
    fn eq(&self, rhs: &Self) -> bool {
        f32_cmp(self.r, rhs.r) &&
        f32_cmp(self.g, rhs.g) &&
        f32_cmp(self.b, rhs.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ops() {
        let c1 = Color::from(0.9f32, 0.6f32, 0.75f32);
        let c2 = Color::from(0.7f32, 0.1f32, 0.25f32);
        let mut c3 = c1.clone();
        let exp = Color::from(1.6f32, 0.7f32, 1.0f32);

        assert_eq!(c1 + c2, exp);

        c3 += c2;
        assert_eq!(c3, exp);
    }

    #[test]
    fn sub_ops() {
        let c1 = Color::from(0.9f32, 0.6f32, 0.75f32);
        let c2 = Color::from(0.7f32, 0.1f32, 0.25f32);
        let mut c3 = c1.clone();
        let exp = Color::from(0.2f32, 0.5f32, 0.5f32);

        assert_eq!(c1 - c2, exp);

        c3 -= c2;
        assert_eq!(c3, exp);
    }

    #[test]
    fn mul_ops() {
        let c1 = Color::from(0.2f32, 0.3f32, 0.4f32);
        let mut c2 = c1.clone();

        let exp1 = Color::from(0.4f32, 0.6f32, 0.8f32);

        assert_eq!(c1 * 2f32, exp1);
        
        c2 *= 2f32;
        assert_eq!(c2, exp1);


        let c3 = Color::from(1f32, 0.2f32, 0.4f32);
        let mut c4 = c3.clone();
        let c5 = Color::from(0.9f32, 1f32, 0.1f32);
        let mut c6 = c5.clone();

        let exp2 = Color::from(0.9f32, 0.2f32, 0.04f32);

        assert_eq!(c3 * c5, exp2);

        c4 *= c5;
        assert_eq!(c4, exp2);

        c6 *= c3;
        assert_eq!(c6, exp2);
    }
}
