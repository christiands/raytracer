use cds_rt::canvas::{Canvas, Coordinate};
use cds_rt::color::Color;

use std::fs::File;
use std::path::Path;

fn main() {
    let mut c = Canvas::new(128, 128).unwrap();

    for x in 0..128 {
        for y in 0..128 {
            let color = Color::from_rgb8(x * 2, y * 2, 0);
            let pixel = Coordinate::from(x as u32, y as u32);
            c.set_pixel(pixel, color).unwrap();
        }
    }

    let path = Path::new("image.png");
    let file = File::create(path).unwrap();

    c.to_png(file).unwrap();
}
