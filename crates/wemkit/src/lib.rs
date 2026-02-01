//! WEM audio conversion library
//!
//! This library provides functionality for:
//! - Decoding WEM files to OGG or WAV (using ww2ogg + symphonia)
//! - Encoding audio files to WEM (using Wwise Console)
//!
//! # Decoding
//!
//! The [`decode`] function converts WEM files to OGG or WAV format.
//! Output format is auto-detected from the file extension.
//!
//! ```no_run
//! use wemkit::{decode, CodebookStrategy};
//! use std::path::PathBuf;
//!
//! // Decode to OGG
//! decode(
//!     &PathBuf::from("input.wem"),
//!     &PathBuf::from("output.ogg"),
//!     CodebookStrategy::Auto,
//!     false,
//!     false,
//! ).unwrap();
//!
//! // Decode to WAV (detected from .wav extension)
//! decode(
//!     &PathBuf::from("input.wem"),
//!     &PathBuf::from("output.wav"),
//!     CodebookStrategy::Auto,
//!     false,
//!     false,
//! ).unwrap();
//! ```
//!
//! # Encoding
//!
//! The [`encode`] function converts audio files (WAV, OGG, MP3, FLAC, AAC) to WEM.
//! Requires Wwise to be installed.
//!
//! ```no_run
//! use wemkit::{encode, EncodeConfig, VorbisQuality};
//! use std::path::PathBuf;
//!
//! let config = EncodeConfig {
//!     input: PathBuf::from("input.wav"),
//!     output: PathBuf::from("output.wem"),
//!     wwise_console: None,
//!     quality: VorbisQuality::High,
//!     sample_rate: None,
//!     channels: None,
//!     volume: None,
//!     verbose: false,
//! };
//! encode(config).unwrap();
//! ```

mod audio;
mod decode;
mod encode;

pub use decode::{CodebookStrategy, ConversionAttempt, decode};
pub use encode::{EncodeConfig, VorbisQuality, encode};
