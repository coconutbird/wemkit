//! Audio processing utilities (format conversion, resampling, etc.)

use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::conv::FromSample;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// ============================================================================
// Standard Downmix Coefficients
// ============================================================================
//
// These coefficients are based on industry standards:
// - ITU-R BS.775-3 for 5.1 surround
// - Dolby/DTS recommendations for 7.1
//
// 5.1 channel order: FL, FR, C, LFE, SL, SR (standard SMPTE/ITU order)
// 7.1 channel order: FL, FR, C, LFE, SL, SR, BL, BR
//
// Standard coefficients for 5.1 → Stereo (ITU-R BS.775):
// - Front L/R: 1.0
// - Center: 1/√2 ≈ 0.707 (split equally to both channels)
// - LFE: Usually 0 or optionally mixed at reduced level
// - Surround L/R: 1/√2 ≈ 0.707
//
// The matrices below use these standard values.

/// Standard downmix coefficient: 1/√2 ≈ 0.7071
const SQRT2_INV: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Standard downmix coefficient: 1/2
const HALF: f32 = 0.5;

/// Common audio channel layouts with known channel ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single channel (mono)
    Mono,
    /// Left, Right
    Stereo,
    /// FL, FR, RL, RR (quadraphonic)
    Quad,
    /// FL, FR, C, LFE, SL, SR (5.1 surround - SMPTE/ITU order)
    Surround51,
    /// FL, FR, C, LFE, SL, SR, BL, BR (7.1 surround)
    Surround71,
    /// Unknown layout with N channels
    Unknown(usize),
}

#[allow(dead_code)]
impl ChannelLayout {
    /// Detect channel layout from channel count
    pub fn from_channel_count(count: usize) -> Self {
        match count {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            4 => ChannelLayout::Quad,
            6 => ChannelLayout::Surround51,
            8 => ChannelLayout::Surround71,
            n => ChannelLayout::Unknown(n),
        }
    }

    /// Get the number of channels in this layout
    pub fn channel_count(&self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Quad => 4,
            ChannelLayout::Surround51 => 6,
            ChannelLayout::Surround71 => 8,
            ChannelLayout::Unknown(n) => *n,
        }
    }
}

/// A downmix matrix that defines how to mix source channels to target channels.
///
/// Each row represents a target channel, each column a source channel.
/// The value at (target, source) is the coefficient to apply.
pub struct DownmixMatrix {
    /// Coefficients: `coefficients[target_channel][source_channel]`
    pub coefficients: Vec<Vec<f32>>,
    /// Number of source channels
    pub source_channels: usize,
    /// Number of target channels
    pub target_channels: usize,
}

impl DownmixMatrix {
    /// Create a new downmix matrix
    pub fn new(coefficients: Vec<Vec<f32>>) -> Self {
        let target_channels = coefficients.len();
        let source_channels = coefficients.first().map(|r| r.len()).unwrap_or(0);
        Self {
            coefficients,
            source_channels,
            target_channels,
        }
    }

    /// Get the standard matrix for 5.1 → Stereo downmix (ITU-R BS.775)
    ///
    /// Channel order: FL, FR, C, LFE, SL, SR
    ///
    /// Formula:
    /// - Left  = FL + (C × 0.707) + (SL × 0.707)
    /// - Right = FR + (C × 0.707) + (SR × 0.707)
    ///
    /// Note: LFE is typically not mixed into stereo output
    pub fn surround51_to_stereo() -> Self {
        Self::new(vec![
            // Left output: FL=1.0, FR=0, C=0.707, LFE=0, SL=0.707, SR=0
            vec![1.0, 0.0, SQRT2_INV, 0.0, SQRT2_INV, 0.0],
            // Right output: FL=0, FR=1.0, C=0.707, LFE=0, SL=0, SR=0.707
            vec![0.0, 1.0, SQRT2_INV, 0.0, 0.0, SQRT2_INV],
        ])
    }

    /// Get the standard matrix for 5.1 → Mono downmix
    ///
    /// Channel order: FL, FR, C, LFE, SL, SR
    ///
    /// Formula: Mono = (FL + FR) × 0.5 + C × 0.707 + (SL + SR) × 0.354
    pub fn surround51_to_mono() -> Self {
        Self::new(vec![
            // Mono output: mix all channels appropriately
            vec![
                HALF,
                HALF,
                SQRT2_INV,
                0.0,
                HALF * SQRT2_INV,
                HALF * SQRT2_INV,
            ],
        ])
    }

    /// Get the standard matrix for 7.1 → Stereo downmix
    ///
    /// Channel order: FL, FR, C, LFE, SL, SR, BL, BR
    ///
    /// Formula:
    /// - Left  = FL + (C × 0.707) + (SL × 0.707) + (BL × 0.5)
    /// - Right = FR + (C × 0.707) + (SR × 0.707) + (BR × 0.5)
    pub fn surround71_to_stereo() -> Self {
        Self::new(vec![
            // Left output
            vec![1.0, 0.0, SQRT2_INV, 0.0, SQRT2_INV, 0.0, HALF, 0.0],
            // Right output
            vec![0.0, 1.0, SQRT2_INV, 0.0, 0.0, SQRT2_INV, 0.0, HALF],
        ])
    }

    /// Get the standard matrix for 7.1 → 5.1 downmix
    ///
    /// Input: FL, FR, C, LFE, SL, SR, BL, BR
    /// Output: FL, FR, C, LFE, SL, SR
    ///
    /// Back channels are mixed into side/surround channels
    pub fn surround71_to_51() -> Self {
        Self::new(vec![
            // FL passthrough
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            // FR passthrough
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            // C passthrough
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            // LFE passthrough
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            // SL = SL + BL × 0.707
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, SQRT2_INV, 0.0],
            // SR = SR + BR × 0.707
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, SQRT2_INV],
        ])
    }

    /// Get the standard matrix for 7.1 → Mono downmix
    ///
    /// Channel order: FL, FR, C, LFE, SL, SR, BL, BR
    pub fn surround71_to_mono() -> Self {
        Self::new(vec![vec![
            HALF,             // FL
            HALF,             // FR
            SQRT2_INV,        // C
            0.0,              // LFE
            HALF * SQRT2_INV, // SL
            HALF * SQRT2_INV, // SR
            HALF * HALF,      // BL
            HALF * HALF,      // BR
        ]])
    }

    /// Get the standard matrix for Quad → Stereo downmix
    ///
    /// Channel order: FL, FR, RL, RR
    ///
    /// Formula:
    /// - Left  = FL + RL × 0.707
    /// - Right = FR + RR × 0.707
    pub fn quad_to_stereo() -> Self {
        Self::new(vec![
            // Left output: FL + RL × 0.707
            vec![1.0, 0.0, SQRT2_INV, 0.0],
            // Right output: FR + RR × 0.707
            vec![0.0, 1.0, 0.0, SQRT2_INV],
        ])
    }

    /// Get the standard matrix for Quad → Mono downmix
    ///
    /// Channel order: FL, FR, RL, RR
    pub fn quad_to_mono() -> Self {
        Self::new(vec![
            // Average front, reduce rear
            vec![HALF, HALF, HALF * SQRT2_INV, HALF * SQRT2_INV],
        ])
    }

    /// Get the standard matrix for Stereo → Mono downmix
    ///
    /// Simple average of left and right
    pub fn stereo_to_mono() -> Self {
        Self::new(vec![vec![HALF, HALF]])
    }

    /// Try to get the appropriate standard matrix for a conversion
    pub fn get_standard_matrix(
        source_layout: ChannelLayout,
        target_layout: ChannelLayout,
    ) -> Option<Self> {
        match (source_layout, target_layout) {
            (ChannelLayout::Stereo, ChannelLayout::Mono) => Some(Self::stereo_to_mono()),
            (ChannelLayout::Quad, ChannelLayout::Mono) => Some(Self::quad_to_mono()),
            (ChannelLayout::Quad, ChannelLayout::Stereo) => Some(Self::quad_to_stereo()),
            (ChannelLayout::Surround51, ChannelLayout::Mono) => Some(Self::surround51_to_mono()),
            (ChannelLayout::Surround51, ChannelLayout::Stereo) => {
                Some(Self::surround51_to_stereo())
            }
            (ChannelLayout::Surround71, ChannelLayout::Mono) => Some(Self::surround71_to_mono()),
            (ChannelLayout::Surround71, ChannelLayout::Stereo) => {
                Some(Self::surround71_to_stereo())
            }
            (ChannelLayout::Surround71, ChannelLayout::Surround51) => {
                Some(Self::surround71_to_51())
            }
            _ => None,
        }
    }

    /// Apply this matrix to downmix audio samples
    ///
    /// Input samples should be interleaved with `source_channels` channels per frame.
    /// Output will be interleaved with `target_channels` channels per frame.
    pub fn apply(&self, samples: &[f32]) -> Vec<f32> {
        let num_frames = samples.len() / self.source_channels;
        let mut output = Vec::with_capacity(num_frames * self.target_channels);

        for frame_idx in 0..num_frames {
            let frame_start = frame_idx * self.source_channels;
            let frame = &samples[frame_start..frame_start + self.source_channels];

            for target_ch in 0..self.target_channels {
                let mut sample = 0.0f32;
                for (source_ch, &coeff) in self.coefficients[target_ch].iter().enumerate() {
                    sample += frame[source_ch] * coeff;
                }
                output.push(sample);
            }
        }

        output
    }
}

/// Convert any supported audio format to WAV using symphonia + hound + rubato
pub fn convert_to_wav(
    input: &Path,
    output: &Path,
    target_sample_rate: Option<u32>,
    channels: Option<u8>,
    volume: Option<f32>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        println!("  Decoding {} with symphonia...", input.display());
    }

    // Open the input file
    let file = File::open(input)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a hint to help the format probe
    let mut hint = Hint::new();
    if let Some(ext) = input.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the format
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    // Get the default track
    let track = format.default_track().ok_or("No audio track found")?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate.ok_or("Unknown sample rate")?;
    let source_channels = codec_params
        .channels
        .ok_or("Unknown channel count")?
        .count();

    if verbose {
        println!("  Source: {}Hz, {} channels", sample_rate, source_channels);
    }

    // Create decoder
    let mut decoder =
        symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

    // Collect all samples as f32
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;

        // Convert to f32 samples (interleaved)
        match decoded {
            AudioBufferRef::F32(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(buf.chan(ch)[frame]);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            AudioBufferRef::S32(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            AudioBufferRef::U8(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            _ => return Err("Unsupported sample format".into()),
        }
    }

    if verbose {
        println!("  Decoded {} samples", all_samples.len());
    }

    // Apply volume adjustment
    if let Some(vol) = volume {
        if verbose {
            println!("  Applying volume: {}x", vol);
        }
        apply_volume(&mut all_samples, vol);
    }

    // Apply channel conversion
    let (output_channels, output_samples) = match channels {
        Some(target_ch) if target_ch as usize != source_channels => {
            let target = target_ch as usize;
            let source_layout = ChannelLayout::from_channel_count(source_channels);
            let target_layout = ChannelLayout::from_channel_count(target);

            if verbose {
                if target < source_channels {
                    println!("  Downmixing {} -> {} channels", source_channels, target);
                } else {
                    println!("  Upmixing {} -> {} channels", source_channels, target);
                }
            }

            // Check if we have a standard matrix for this conversion
            let has_standard_matrix =
                DownmixMatrix::get_standard_matrix(source_layout, target_layout).is_some();

            // Warn only for non-standard conversions where we fall back to simple mixing
            if target < source_channels && !has_standard_matrix {
                eprintln!(
                    "  ⚠ Warning: Channel conversion {}->{} uses simple averaging. \
                     No standard downmix matrix available for this layout.",
                    source_channels, target
                );
            }
            (
                target,
                convert_channels(&all_samples, source_channels, target),
            )
        }
        Some(ch) => (ch as usize, all_samples),
        None => (source_channels, all_samples),
    };

    // Apply sample rate conversion if requested
    let (final_sample_rate, final_samples) = if let Some(target_sr) = target_sample_rate {
        if target_sr != sample_rate {
            if verbose {
                println!("  Resampling {}Hz -> {}Hz", sample_rate, target_sr);
            }

            use audioadapter_buffers::direct::SequentialSliceOfVecs;
            use rubato::{Fft, FixedSync, Resampler};

            // Convert interleaved samples to separate channels
            let num_frames = output_samples.len() / output_channels;
            let mut channel_data: Vec<Vec<f32>> = (0..output_channels)
                .map(|_| Vec::with_capacity(num_frames))
                .collect();

            for (i, sample) in output_samples.iter().enumerate() {
                channel_data[i % output_channels].push(*sample);
            }

            // Create resampler
            let mut resampler = Fft::<f32>::new(
                target_sr as usize,
                sample_rate as usize,
                1024,
                2, // sub-chunks
                output_channels,
                FixedSync::Input,
            )?;

            // Calculate output size and create output buffer
            let output_frames = resampler.process_all_needed_output_len(num_frames);
            let mut output_buffer: Vec<Vec<f32>> = (0..output_channels)
                .map(|_| vec![0.0f32; output_frames])
                .collect();

            // Wrap input and output with adapters
            let input_adapter =
                SequentialSliceOfVecs::new(&channel_data, output_channels, num_frames)?;
            let mut output_adapter =
                SequentialSliceOfVecs::new_mut(&mut output_buffer, output_channels, output_frames)?;

            // Process all at once
            let (_frames_read, frames_written) = resampler.process_all_into_buffer(
                &input_adapter,
                &mut output_adapter,
                num_frames,
                None,
            )?;

            // Convert back to interleaved
            let mut interleaved = Vec::with_capacity(frames_written * output_channels);
            for frame in 0..frames_written {
                for ch in &output_buffer {
                    interleaved.push(ch[frame]);
                }
            }

            (target_sr, interleaved)
        } else {
            (sample_rate, output_samples)
        }
    } else {
        (sample_rate, output_samples)
    };

    // Write WAV file using hound
    let spec = hound::WavSpec {
        channels: output_channels as u16,
        sample_rate: final_sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    if verbose {
        println!(
            "  Writing WAV: {}Hz, {} channels",
            final_sample_rate, output_channels
        );
    }

    let mut writer = hound::WavWriter::create(output, spec)?;

    for sample in final_samples {
        // Convert f32 [-1.0, 1.0] to i16
        let s16 = (sample * 32767.0) as i16;
        writer.write_sample(s16)?;
    }

    writer.finalize()?;

    if verbose {
        println!("  WAV written to {}", output.display());
    }

    Ok(())
}

/// Convert OGG audio bytes in memory to a WAV file.
///
/// This function takes raw OGG Vorbis data as bytes and writes a WAV file,
/// avoiding any intermediate files on disk. Used by the decode function
/// when outputting to WAV format.
///
/// # Arguments
///
/// * `ogg_data` - Raw OGG Vorbis audio data
/// * `output` - Path to write the output WAV file
/// * `verbose` - Print detailed progress information
pub fn convert_ogg_bytes_to_wav(
    ogg_data: &[u8],
    output: &Path,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        println!("  Decoding OGG from memory with symphonia...");
    }

    // Create a media source from the in-memory OGG data
    let cursor = Cursor::new(ogg_data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    // Create a hint for OGG format
    let mut hint = Hint::new();
    hint.with_extension("ogg");

    // Probe the format
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    // Get the default track
    let track = format.default_track().ok_or("No audio track found")?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate.ok_or("Unknown sample rate")?;
    let channels = codec_params
        .channels
        .ok_or("Unknown channel count")?
        .count();

    if verbose {
        println!("  Source: {}Hz, {} channels", sample_rate, channels);
    }

    // Create decoder
    let mut decoder =
        symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

    // Collect all samples as f32
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;

        // Convert to f32 samples (interleaved)
        match decoded {
            AudioBufferRef::F32(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(buf.chan(ch)[frame]);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            AudioBufferRef::S32(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            AudioBufferRef::U8(buf) => {
                for frame in 0..buf.frames() {
                    for ch in 0..buf.spec().channels.count() {
                        all_samples.push(f32::from_sample(buf.chan(ch)[frame]));
                    }
                }
            }
            _ => return Err("Unsupported sample format".into()),
        }
    }

    // Write WAV file using hound
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    if verbose {
        println!("  Writing WAV: {}Hz, {} channels", sample_rate, channels);
    }

    let mut writer = hound::WavWriter::create(output, spec)?;

    for sample in all_samples {
        // Convert f32 [-1.0, 1.0] to i16
        let s16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(s16)?;
    }

    writer.finalize()?;

    if verbose {
        println!("  WAV written to {}", output.display());
    }

    Ok(())
}

/// Apply volume adjustment to samples (clamped to [-1.0, 1.0])
pub(crate) fn apply_volume(samples: &mut [f32], volume: f32) {
    for sample in samples {
        *sample *= volume;
        *sample = sample.clamp(-1.0, 1.0);
    }
}

/// Downmix multi-channel audio to fewer channels.
///
/// Uses standard downmix matrices (ITU-R BS.775) for known surround formats:
/// - 5.1 → Stereo: ITU-R BS.775 coefficients
/// - 7.1 → Stereo: Standard surround coefficients
/// - Quad → Stereo: Rear channels attenuated by 3dB
///
/// For unknown layouts, falls back to simple averaging.
pub(crate) fn downmix(samples: &[f32], source_channels: usize, target_channels: usize) -> Vec<f32> {
    let source_layout = ChannelLayout::from_channel_count(source_channels);
    let target_layout = ChannelLayout::from_channel_count(target_channels);

    // Try to use a standard matrix for known layouts
    if let Some(matrix) = DownmixMatrix::get_standard_matrix(source_layout, target_layout) {
        return matrix.apply(samples);
    }

    // Fallback: simple averaging for unknown layouts
    downmix_simple(samples, source_channels, target_channels)
}

/// Simple downmix by averaging (fallback for unknown layouts)
/// - To mono: averages all channels
/// - To N channels: groups source channels and averages each group
fn downmix_simple(samples: &[f32], source_channels: usize, target_channels: usize) -> Vec<f32> {
    if target_channels == 1 {
        // Special case: mono - average all channels
        samples
            .chunks(source_channels)
            .map(|frame| frame.iter().sum::<f32>() / source_channels as f32)
            .collect()
    } else {
        // General case: divide source channels into target_channels groups
        samples
            .chunks(source_channels)
            .flat_map(|frame| {
                (0..target_channels).map(|target_ch| {
                    // Calculate which source channels map to this target channel
                    let start = target_ch * source_channels / target_channels;
                    let end = (target_ch + 1) * source_channels / target_channels;
                    let count = end - start;
                    frame[start..end].iter().sum::<f32>() / count as f32
                })
            })
            .collect()
    }
}

/// Upmix audio to more channels by cycling through source channels
/// - Mono: duplicate to all channels
/// - Stereo to 6: L, R, L, R, L, R
/// - General: cycles source channels to fill target
pub(crate) fn upmix(samples: &[f32], source_channels: usize, target_channels: usize) -> Vec<f32> {
    samples
        .chunks(source_channels)
        .flat_map(|frame| (0..target_channels).map(|ch| frame[ch % source_channels]))
        .collect()
}

/// Convert channels - handles both upmixing and downmixing
pub(crate) fn convert_channels(
    samples: &[f32],
    source_channels: usize,
    target_channels: usize,
) -> Vec<f32> {
    match target_channels.cmp(&source_channels) {
        std::cmp::Ordering::Less => downmix(samples, source_channels, target_channels),
        std::cmp::Ordering::Greater => upmix(samples, source_channels, target_channels),
        std::cmp::Ordering::Equal => samples.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_volume_increase() {
        let mut samples = vec![0.5, -0.5, 0.25, -0.25];
        apply_volume(&mut samples, 1.5);
        assert_eq!(samples, vec![0.75, -0.75, 0.375, -0.375]);
    }

    #[test]
    fn test_apply_volume_decrease() {
        let mut samples = vec![1.0, -1.0, 0.5, -0.5];
        apply_volume(&mut samples, 0.5);
        assert_eq!(samples, vec![0.5, -0.5, 0.25, -0.25]);
    }

    #[test]
    fn test_apply_volume_clamps_positive() {
        let mut samples = vec![0.8, 0.9, 1.0];
        apply_volume(&mut samples, 2.0);
        // All should be clamped to 1.0
        assert_eq!(samples, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_apply_volume_clamps_negative() {
        let mut samples = vec![-0.8, -0.9, -1.0];
        apply_volume(&mut samples, 2.0);
        // All should be clamped to -1.0
        assert_eq!(samples, vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_downmix_stereo_to_mono() {
        // Stereo: [L, R, L, R, L, R]
        let stereo = vec![1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        let mono = downmix(&stereo, 2, 1);
        // Expected: average of each pair
        assert_eq!(mono, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_downmix_stereo_to_mono_identical_channels() {
        let stereo = vec![0.5, 0.5, -0.25, -0.25];
        let mono = downmix(&stereo, 2, 1);
        assert_eq!(mono, vec![0.5, -0.25]);
    }

    #[test]
    fn test_downmix_quad_to_mono() {
        // 4 channels: [FL, FR, RL, RR] per frame
        // Using standard matrix: FL*0.5 + FR*0.5 + RL*0.354 + RR*0.354
        let quad = vec![1.0, 1.0, 1.0, 1.0];
        let mono = downmix(&quad, 4, 1);
        // Expected: 0.5 + 0.5 + 0.354 + 0.354 ≈ 1.707
        let expected = HALF + HALF + HALF * SQRT2_INV + HALF * SQRT2_INV;
        assert!((mono[0] - expected).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_quad_to_stereo() {
        // 4 channels: [FL, FR, RL, RR] per frame
        // Using standard matrix: L = FL + RL*0.707, R = FR + RR*0.707
        let quad = vec![1.0, 0.5, 0.5, 0.25];
        let stereo = downmix(&quad, 4, 2);
        // Left: 1.0 + 0.5 * 0.707 ≈ 1.354
        // Right: 0.5 + 0.25 * 0.707 ≈ 0.677
        let expected_l = 1.0 + 0.5 * SQRT2_INV;
        let expected_r = 0.5 + 0.25 * SQRT2_INV;
        assert!((stereo[0] - expected_l).abs() < 0.0001);
        assert!((stereo[1] - expected_r).abs() < 0.0001);
    }

    #[test]
    fn test_upmix_mono_to_stereo() {
        let mono = vec![0.5, -0.25, 1.0];
        let stereo = upmix(&mono, 1, 2);
        // Each sample duplicated
        assert_eq!(stereo, vec![0.5, 0.5, -0.25, -0.25, 1.0, 1.0]);
    }

    #[test]
    fn test_upmix_stereo_to_quad() {
        // Stereo [L, R] -> Quad [L, R, L, R]
        let stereo = vec![1.0, 0.5];
        let quad = upmix(&stereo, 2, 4);
        assert_eq!(quad, vec![1.0, 0.5, 1.0, 0.5]);
    }

    #[test]
    fn test_upmix_mono_to_six_channels() {
        let mono = vec![0.5];
        let six = upmix(&mono, 1, 6);
        assert_eq!(six, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_convert_channels_same() {
        let stereo = vec![1.0, 0.5, -1.0, -0.5];
        let result = convert_channels(&stereo, 2, 2);
        assert_eq!(result, stereo);
    }

    #[test]
    fn test_convert_channels_downmix() {
        let stereo = vec![1.0, 0.0];
        let mono = convert_channels(&stereo, 2, 1);
        assert_eq!(mono, vec![0.5]);
    }

    #[test]
    fn test_convert_channels_upmix() {
        let mono = vec![0.5];
        let stereo = convert_channels(&mono, 1, 2);
        assert_eq!(stereo, vec![0.5, 0.5]);
    }

    #[test]
    fn test_upmix_empty() {
        let samples: Vec<f32> = vec![];
        let result = upmix(&samples, 1, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downmix_empty() {
        let samples: Vec<f32> = vec![];
        let mono = downmix(&samples, 2, 1);
        assert!(mono.is_empty());
    }

    #[test]
    fn test_roundtrip_mono_stereo_mono() {
        let original = vec![0.5, -0.25, 0.75];
        let stereo = upmix(&original, 1, 2);
        let back_to_mono = downmix(&stereo, 2, 1);
        assert_eq!(original, back_to_mono);
    }

    // ========================================================================
    // Downmix Matrix Tests
    // ========================================================================

    #[test]
    fn test_channel_layout_from_count() {
        assert_eq!(ChannelLayout::from_channel_count(1), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from_channel_count(2), ChannelLayout::Stereo);
        assert_eq!(ChannelLayout::from_channel_count(4), ChannelLayout::Quad);
        assert_eq!(
            ChannelLayout::from_channel_count(6),
            ChannelLayout::Surround51
        );
        assert_eq!(
            ChannelLayout::from_channel_count(8),
            ChannelLayout::Surround71
        );
        assert_eq!(
            ChannelLayout::from_channel_count(3),
            ChannelLayout::Unknown(3)
        );
    }

    #[test]
    fn test_channel_layout_channel_count() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Quad.channel_count(), 4);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayout::Unknown(5).channel_count(), 5);
    }

    #[test]
    fn test_downmix_51_to_stereo() {
        // 5.1 channel order: FL, FR, C, LFE, SL, SR
        // Test with unit signals to verify coefficients
        let surround51 = vec![1.0, 0.0, 1.0, 0.5, 1.0, 0.0]; // FL=1, FR=0, C=1, LFE=0.5, SL=1, SR=0
        let stereo = downmix(&surround51, 6, 2);
        // Left = FL + C*0.707 + SL*0.707 = 1.0 + 0.707 + 0.707 ≈ 2.414
        // Right = FR + C*0.707 + SR*0.707 = 0.0 + 0.707 + 0.0 ≈ 0.707
        let expected_l = 1.0 + SQRT2_INV + SQRT2_INV;
        let expected_r = 0.0 + SQRT2_INV + 0.0;
        assert!((stereo[0] - expected_l).abs() < 0.0001);
        assert!((stereo[1] - expected_r).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_51_to_stereo_lfe_ignored() {
        // Verify LFE channel is not mixed into stereo
        let surround51 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // Only LFE = 1.0
        let stereo = downmix(&surround51, 6, 2);
        // Both channels should be 0 since LFE is not mixed
        assert!((stereo[0]).abs() < 0.0001);
        assert!((stereo[1]).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_51_to_mono() {
        // 5.1: FL, FR, C, LFE, SL, SR - all at 1.0 except LFE
        let surround51 = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let mono = downmix(&surround51, 6, 1);
        // Mono = FL*0.5 + FR*0.5 + C*0.707 + SL*0.354 + SR*0.354
        let expected = HALF + HALF + SQRT2_INV + HALF * SQRT2_INV + HALF * SQRT2_INV;
        assert!((mono[0] - expected).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_71_to_stereo() {
        // 7.1 channel order: FL, FR, C, LFE, SL, SR, BL, BR
        let surround71 = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let stereo = downmix(&surround71, 8, 2);
        // Left = FL + C*0.707 + SL*0.707 + BL*0.5 = 1.0 + 0.707 + 0.707 + 0.5 ≈ 2.914
        // Right = FR + C*0.707 + SR*0.707 + BR*0.5 = 0.0 + 0.707 + 0.0 + 0.0 ≈ 0.707
        let expected_l = 1.0 + SQRT2_INV + SQRT2_INV + HALF;
        let expected_r = 0.0 + SQRT2_INV + 0.0 + 0.0;
        assert!((stereo[0] - expected_l).abs() < 0.0001);
        assert!((stereo[1] - expected_r).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_71_to_51() {
        // 7.1 -> 5.1: back channels fold into surround
        let surround71 = vec![1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4, 0.6];
        let surround51 = downmix(&surround71, 8, 6);
        // FL, FR, C, LFE pass through
        assert!((surround51[0] - 1.0).abs() < 0.0001); // FL
        assert!((surround51[1] - 0.5).abs() < 0.0001); // FR
        assert!((surround51[2] - 0.25).abs() < 0.0001); // C
        assert!((surround51[3] - 0.1).abs() < 0.0001); // LFE
        // SL = SL + BL*0.707 = 0.2 + 0.4*0.707
        let expected_sl = 0.2 + 0.4 * SQRT2_INV;
        assert!((surround51[4] - expected_sl).abs() < 0.0001);
        // SR = SR + BR*0.707 = 0.3 + 0.6*0.707
        let expected_sr = 0.3 + 0.6 * SQRT2_INV;
        assert!((surround51[5] - expected_sr).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_matrix_get_standard() {
        // Test that standard matrices are returned for known conversions
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Stereo, ChannelLayout::Mono)
                .is_some()
        );
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Quad, ChannelLayout::Stereo)
                .is_some()
        );
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Surround51, ChannelLayout::Stereo)
                .is_some()
        );
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Surround71, ChannelLayout::Stereo)
                .is_some()
        );
        assert!(
            DownmixMatrix::get_standard_matrix(
                ChannelLayout::Surround71,
                ChannelLayout::Surround51
            )
            .is_some()
        );

        // Test that None is returned for unknown conversions
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Unknown(5), ChannelLayout::Stereo)
                .is_none()
        );
        assert!(
            DownmixMatrix::get_standard_matrix(ChannelLayout::Stereo, ChannelLayout::Quad)
                .is_none()
        );
    }

    #[test]
    fn test_downmix_unknown_layout_uses_fallback() {
        // 3 channels -> 1 channel should use simple averaging (no standard matrix)
        let three_ch = vec![0.6, 0.3, 0.9];
        let mono = downmix(&three_ch, 3, 1);
        // Simple average: (0.6 + 0.3 + 0.9) / 3 = 0.6
        assert!((mono[0] - 0.6).abs() < 0.0001);
    }

    #[test]
    fn test_downmix_matrix_apply_multiple_frames() {
        // Test that matrix correctly processes multiple frames
        let matrix = DownmixMatrix::stereo_to_mono();
        let stereo = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let mono = matrix.apply(&stereo);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 0.0001); // (1.0 + 0.0) / 2
        assert!((mono[1] - 0.5).abs() < 0.0001); // (0.0 + 1.0) / 2
        assert!((mono[2] - 0.5).abs() < 0.0001); // (0.5 + 0.5) / 2
    }
}
