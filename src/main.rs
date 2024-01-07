use std::{path::Path, sync::Arc};

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f32::consts::PI;
use wave_stream::{
    samples_by_channel::SamplesByChannel,
    wave_header::{Channels, SampleFormat, WavHeader},
    wave_writer::RandomAccessWavWriter,
    write_wav_to_file_path,
};

const HALF_PI: f32 = PI / 2.0;

const WINDOW_SIZE: usize = 50;
const ITERATIONS_PER_TONE: usize = 200;
const ITERATIONS_PER_SILENCE: usize = 20;

struct ToneGenerator {
    header: WavHeader,
    window_size: usize,
    iterations_per_tone: usize,
    iterations_per_silence: usize,
    fft_inverse: Arc<dyn Fft<f32>>,
    scale: f32,
    scratch: Vec<Complex<f32>>,

    sample_ctr: usize,
}

fn main() {
    println!("Generating test tones for use with soft_matrix");
    println!();
    println!("Tones are always in the order:");
    println!("\tcenter");
    println!("\tright front");
    println!("\tright middle");
    println!("\tright rear");
    println!("\trear center");
    println!("\tleft rear");
    println!("\tleft middle");
    println!("\tleft front");

    let sample_rate = 44100;
    let header = WavHeader {
        sample_format: SampleFormat::Float,
        channels: Channels::new().front_left().front_right(),
        sample_rate,
    };

    let mut planner = FftPlanner::new();
    let fft_inverse = planner.plan_fft_inverse(WINDOW_SIZE);

    let scratch = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        fft_inverse.get_inplace_scratch_len()
    ];

    let mut tone_generator = ToneGenerator {
        header,
        window_size: WINDOW_SIZE,
        iterations_per_tone: ITERATIONS_PER_TONE,
        iterations_per_silence: ITERATIONS_PER_SILENCE,
        fft_inverse,

        // rustfft states that the scale is 1/len()
        // See "noramlization": https://docs.rs/rustfft/latest/rustfft/#normalization
        scale: 1.0 / (WINDOW_SIZE as f32).sqrt(),

        scratch,
        sample_ctr: 0,
    };

    // default
    tone_generator.write_all_tones(
        Path::new("default.wav"),
        (
            Complex::from_polar(0.707106781186548, 0.0),
            Complex::from_polar(0.707106781186548, 0.0),
        ),
        (Complex::from_polar(0.0, 0.0), Complex::from_polar(1.0, 0.0)),
        (
            Complex::from_polar(0.1, HALF_PI),
            Complex::from_polar(1.0, 0.0),
        ),
        (Complex::from_polar(0.1, PI), Complex::from_polar(1.0, 0.0)),
        (
            Complex::from_polar(0.707106781186548, PI),
            Complex::from_polar(0.707106781186548, 0.0),
        ),
        (Complex::from_polar(1.0, PI), Complex::from_polar(0.1, 0.0)),
        (
            Complex::from_polar(1.0, PI),
            Complex::from_polar(0.1, HALF_PI),
        ),
        (Complex::from_polar(1.0, 0.0), Complex::from_polar(0.0, 0.0)),
    );

    let (right_middle_lt, right_middle_rt) = sq_encode(
        Complex::from_polar(0.0, 0.0),
        Complex::from_polar(0.707106781186548, 0.0),
        Complex::from_polar(0.0, 0.0),
        Complex::from_polar(0.707106781186548, 0.0),
    );

    let (rear_center_lt, rear_center_rt) = sq_encode(
        Complex::from_polar(0.0, 0.0),
        Complex::from_polar(0.0, 0.0),
        Complex::from_polar(0.707106781186548, 0.0),
        Complex::from_polar(0.707106781186548, 0.0),
    );

    let (left_middle_lt, left_middle_rt) = sq_encode(
        Complex::from_polar(0.707106781186548, 0.0),
        Complex::from_polar(0.0, 0.0),
        Complex::from_polar(0.707106781186548, 0.0),
        Complex::from_polar(0.0, 0.0),
    );

    // sq
    tone_generator.write_all_tones(
        Path::new("sq.wav"),
        (Complex::from_polar(0.707106781186548, 0.0), Complex::from_polar(0.707106781186548, 0.0)),
        (Complex::from_polar(0.0, 0.0), Complex::from_polar(1.0, 0.0)),
        (right_middle_lt, right_middle_rt),
        (
            Complex::from_polar(0.7, 0.0),
            Complex::from_polar(0.7, HALF_PI),
        ),
        (
            Complex::from_polar(0.7, 0.0) + Complex::from_polar(0.7, 0.0 - HALF_PI),
            Complex::from_polar(0.7, HALF_PI) + Complex::from_polar(0.7, PI),
        ),
        (rear_center_lt, rear_center_rt),
        (left_middle_lt, left_middle_rt),
        (Complex::from_polar(1.0, 0.0), Complex::from_polar(0.0, 0.0)),
    );
}

fn sq_encode(
    left_front: Complex<f32>,
    right_front: Complex<f32>,
    left_rear: Complex<f32>,
    right_rear: Complex<f32>,
) -> (Complex<f32>, Complex<f32>) {
    let (left_back_amplitude, left_back_phase) = left_rear.to_polar();
    let (right_back_amplitude, right_back_phase) = right_rear.to_polar();

    let left_back_for_left_total =
        Complex::from_polar(0.7 * left_back_amplitude, left_back_phase - HALF_PI);
    let right_back_for_left_total =
        Complex::from_polar(0.7 * right_back_amplitude, right_back_phase);
    let left_total = left_front + left_back_for_left_total + right_back_for_left_total;

    let left_back_for_right_total =
        Complex::from_polar(0.7 * left_back_amplitude, left_back_phase + PI);
    let right_back_for_right_total =
        Complex::from_polar(0.7 * right_back_amplitude, right_back_phase + HALF_PI);
    let right_total = right_front + left_back_for_right_total + right_back_for_right_total;

    (left_front + left_total, right_front + right_total)
}

impl ToneGenerator {
    pub fn write_all_tones(
        &mut self,
        path: &Path,
        center: (Complex<f32>, Complex<f32>),
        right_front: (Complex<f32>, Complex<f32>),
        right_middle: (Complex<f32>, Complex<f32>),
        right_rear: (Complex<f32>, Complex<f32>),
        rear_center: (Complex<f32>, Complex<f32>),
        left_rear: (Complex<f32>, Complex<f32>),
        left_middle: (Complex<f32>, Complex<f32>),
        left_front: (Complex<f32>, Complex<f32>),
    ) {
        self.sample_ctr = 0;

        let outfile = write_wav_to_file_path(path, self.header).unwrap();
        let mut writer = outfile.get_random_access_f32_writer().unwrap();

        self.write_silence(&mut writer);

        // Center
        self.write_tones(&mut writer, self.create_window(center));
        self.write_silence(&mut writer);

        // Right front
        self.write_tones(&mut writer, self.create_window(right_front));
        self.write_silence(&mut writer);

        // Right middle
        self.write_tones(&mut writer, self.create_window(right_middle));
        self.write_silence(&mut writer);

        // Right rear
        self.write_tones(&mut writer, self.create_window(right_rear));
        self.write_silence(&mut writer);

        // Rear center
        self.write_tones(&mut writer, self.create_window(rear_center));
        self.write_silence(&mut writer);

        // Left rear
        self.write_tones(&mut writer, self.create_window(left_rear));
        self.write_silence(&mut writer);

        // Left middle
        self.write_tones(&mut writer, self.create_window(left_middle));
        self.write_silence(&mut writer);

        // Left front
        self.write_tones(&mut writer, self.create_window(left_front));
        self.write_silence(&mut writer);

        writer.flush().unwrap();
    }

    fn create_window(
        &self,
        tones: (Complex<f32>, Complex<f32>),
    ) -> (Vec<Complex<f32>>, Vec<Complex<f32>>) {
        let (left_total_tone, right_total_tone) = tones;

        let mut right_total_window = vec![Complex::new(0.0, 0.0); self.window_size];
        right_total_window[1] = right_total_tone;
        right_total_window[self.window_size - 1] = Complex {
            re: right_total_tone.re,
            im: -1.0 * right_total_tone.im,
        };

        let mut left_total_window = vec![Complex::new(0.0, 0.0); self.window_size];
        left_total_window[1] = left_total_tone;
        left_total_window[self.window_size - 1] = Complex {
            re: left_total_tone.re,
            im: -1.0 * left_total_tone.im,
        };

        (left_total_window, right_total_window)
    }

    fn write_tones(
        &mut self,
        writer: &mut RandomAccessWavWriter<f32>,
        windows: (Vec<Complex<f32>>, Vec<Complex<f32>>),
    ) {
        let (mut left_total_window, mut right_total_window) = windows;
        self.fft_inverse
            .process_with_scratch(&mut left_total_window, &mut self.scratch);
        self.fft_inverse
            .process_with_scratch(&mut right_total_window, &mut self.scratch);

        for _iteration in 0..self.iterations_per_tone {
            for window_ctr in 0..self.window_size {
                let samples_by_channel = SamplesByChannel::new()
                    .front_left(self.scale * left_total_window[window_ctr].re)
                    .front_right(self.scale * right_total_window[window_ctr].re);

                writer
                    .write_samples(self.sample_ctr, samples_by_channel)
                    .unwrap();

                self.sample_ctr += 1;
            }
        }
    }

    fn write_silence(&mut self, writer: &mut RandomAccessWavWriter<f32>) {
        for _ in 0..self.iterations_per_silence {
            for _ in 0..self.window_size {
                let samples_by_channel = SamplesByChannel::new().front_left(0.0).front_right(0.0);

                writer
                    .write_samples(self.sample_ctr, samples_by_channel)
                    .unwrap();

                self.sample_ctr += 1;
            }
        }
    }
}
