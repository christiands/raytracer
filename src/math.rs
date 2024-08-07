const F32_EPSILON: f32 = 0.00001f32;

pub fn f32_cmp(a: f32, b: f32) -> bool {
    (a - b).abs() < F32_EPSILON
}
