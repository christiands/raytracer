use crate::color::Color;

use std::error::Error;
use std::io::BufWriter;
use std::fs::File;
use png::{Encoder, ColorType, BitDepth};

#[derive(Clone, Copy, Debug)]
pub struct Coordinate(u32, u32);

impl Coordinate {
    pub fn from(x: u32, y: u32) -> Self {
        Coordinate(x, y)
    }
}

pub struct Canvas {
    data: Vec<Color>,
    width: u32, // horizontal axis (x)
    height: u32, // vertical axis (y)
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        Self::with_fill(width, height, Color::new())
    }

    pub fn with_fill(width: u32, height: u32, color: Color) -> Result<Self, Box<dyn Error>> {
        if width == 0 || height == 0 {
            Err("Cannot create a canvas with a side length of 0".into())
        }
        else {
            let mut temp: Vec<Color> = Vec::with_capacity((width * height) as usize);
            temp.resize((width * height) as usize, color);

            Ok(Self {
                data: temp,
                width: width,
                height: height,
            })
        }
    }

    pub fn get_pixel(&self, pixel: Coordinate) -> Option<Color> {
        if !self.in_bounds(pixel) {
            None
        }
        else {
            let index = (self.width * pixel.1 + pixel.0) as usize;
            Some(self.data[index])
        }
    }

    // If "pixel" is out of bounds, returns Err(())
    pub fn set_pixel(&mut self, pixel: Coordinate, color: Color) -> Result<(), ()> {
        if !self.in_bounds(pixel) {
            Err(())
        }
        else {
            let index = (self.width * pixel.1 + pixel.0) as usize;
            self.data[index] = color;
            Ok(())
        }
    }

    pub fn in_bounds(&self, pixel: Coordinate) -> bool {
        pixel.0 < self.width && pixel.1 < self.height
    }

    pub fn dimensions(&self) -> Coordinate {
        Coordinate(self.width, self.height)
    }
    
    pub fn as_u8_vec(&self) -> Vec<u8> {
        let mut temp = Vec::new();

        for color in &self.data {
            temp.extend_from_slice(&color.as_rgb8_bytes());
        }

        temp
    }
    
    pub fn to_png(&self, file: File) -> Result<(), Box<dyn Error>> {
        let mut writer = BufWriter::new(file);
        let mut encoder = Encoder::new(&mut writer, self.width, self.height);

        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);

        let mut png_writer = match encoder.write_header() {
            Ok(w) => w,
            Err(e) => return Err(e.to_string().into()),
        };

        match png_writer.write_image_data(&self.as_u8_vec()[..]) {
            Ok(()) => (),
            Err(e) => return Err(e.to_string().into()),
        };

        Ok(())
    }
}
