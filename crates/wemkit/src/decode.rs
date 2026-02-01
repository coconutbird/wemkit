//! WEM to OGG/WAV decoding functionality

use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

use ww2ogg::{CodebookLibrary, WwiseRiffVorbis, validate};

use crate::audio::convert_ogg_bytes_to_wav;

/// Codebook selection strategy for WEM to OGG conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodebookStrategy {
    /// Automatically detect the correct codebook by trying both and validating
    #[default]
    Auto,
    /// Use the default/packed codebooks
    Default,
    /// Use aoTuV 6.03 codebooks
    Aotuv,
}

impl std::fmt::Display for CodebookStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodebookStrategy::Auto => write!(f, "auto"),
            CodebookStrategy::Default => write!(f, "default"),
            CodebookStrategy::Aotuv => write!(f, "aotuv"),
        }
    }
}

/// Result of a conversion attempt with a specific codebook
#[derive(Debug)]
pub struct ConversionAttempt {
    pub codebook: CodebookStrategy,
    pub data: Vec<u8>,
    pub validation_passed: bool,
}

/// Attempt conversion with a specific codebook, returning the result in memory
fn try_convert_with_codebook(
    input_data: &[u8],
    codebook: CodebookStrategy,
    verbose: bool,
) -> Result<ConversionAttempt, Box<dyn std::error::Error>> {
    let codebooks = match codebook {
        CodebookStrategy::Default | CodebookStrategy::Auto => CodebookLibrary::default_codebooks()?,
        CodebookStrategy::Aotuv => CodebookLibrary::aotuv_codebooks()?,
    };

    let input_cursor = Cursor::new(input_data);
    let mut converter = WwiseRiffVorbis::new(input_cursor, codebooks)?;

    let mut output_buffer = Vec::new();
    converter.generate_ogg(&mut output_buffer)?;

    // Validate the output
    let validation_passed = match validate(&output_buffer) {
        Ok(()) => {
            if verbose {
                println!("  ✓ Validation passed with {} codebook", codebook);
            }
            true
        }
        Err(e) => {
            if verbose {
                println!("  ✗ Validation failed with {} codebook: {}", codebook, e);
            }
            false
        }
    };

    Ok(ConversionAttempt {
        codebook,
        data: output_buffer,
        validation_passed,
    })
}

/// Decode a WEM file to OGG or WAV format.
///
/// The output format is automatically detected from the file extension:
/// - `.ogg` → OGG Vorbis (direct output from ww2ogg)
/// - `.wav` → WAV (converted in-memory via symphonia)
///
/// # Arguments
///
/// * `input` - Path to the input WEM file
/// * `output` - Path to the output file (.ogg or .wav)
/// * `strategy` - Codebook selection strategy for decoding
/// * `skip_validation` - Skip validation of the output OGG data
/// * `verbose` - Print detailed progress information
///
/// # Errors
///
/// Returns an error if:
/// - The input file cannot be read
/// - The WEM file is invalid or uses an unsupported format
/// - The output file cannot be written
pub fn decode(
    input: &Path,
    output: &Path,
    strategy: CodebookStrategy,
    skip_validation: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting {} -> {}", input.display(), output.display());

    // Read input file into memory for potential multiple attempts
    let input_data = std::fs::read(input)?;

    let result = match strategy {
        CodebookStrategy::Auto => {
            println!("Auto-detecting codebook...");

            // Try default codebooks first
            if verbose {
                println!("  Attempting with default codebooks...");
            }
            let default_attempt =
                try_convert_with_codebook(&input_data, CodebookStrategy::Default, verbose);

            match default_attempt {
                Ok(attempt) if attempt.validation_passed || skip_validation => {
                    println!("  → Using default codebooks");
                    attempt
                }
                Ok(_) | Err(_) => {
                    // Try aoTuV codebooks
                    if verbose {
                        println!("  Attempting with aoTuV codebooks...");
                    }
                    let aotuv_attempt =
                        try_convert_with_codebook(&input_data, CodebookStrategy::Aotuv, verbose)?;

                    if aotuv_attempt.validation_passed || skip_validation {
                        println!("  → Using aoTuV codebooks");
                        aotuv_attempt
                    } else {
                        // Both failed validation, return the default attempt with a warning
                        eprintln!(
                            "⚠ Warning: Neither codebook produced valid output. Using default."
                        );
                        try_convert_with_codebook(&input_data, CodebookStrategy::Default, verbose)?
                    }
                }
            }
        }
        CodebookStrategy::Default => {
            println!("Using default codebooks (forced)");
            let attempt =
                try_convert_with_codebook(&input_data, CodebookStrategy::Default, verbose)?;
            if !skip_validation && !attempt.validation_passed {
                eprintln!("⚠ Warning: Validation failed with default codebooks");
            }
            attempt
        }
        CodebookStrategy::Aotuv => {
            println!("Using aoTuV codebooks (forced)");
            let attempt = try_convert_with_codebook(&input_data, CodebookStrategy::Aotuv, verbose)?;
            if !skip_validation && !attempt.validation_passed {
                eprintln!("⚠ Warning: Validation failed with aoTuV codebooks");
            }
            attempt
        }
    };

    // Detect output format from extension
    let output_ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match output_ext.as_deref() {
        Some("wav") => {
            // Convert OGG bytes directly to WAV in memory (no temp files)
            if verbose {
                println!("  Converting OGG to WAV...");
            }

            convert_ogg_bytes_to_wav(&result.data, output, verbose)?;

            let file_size = std::fs::metadata(output)?.len();
            println!(
                "✓ Conversion complete! ({} bytes WAV, codebook: {})",
                file_size, result.codebook
            );
        }
        _ => {
            // Default: write OGG directly (existing behavior)
            let mut output_file = BufWriter::new(File::create(output)?);
            output_file.write_all(&result.data)?;
            output_file.flush()?;

            println!(
                "✓ Conversion complete! ({} bytes, codebook: {})",
                result.data.len(),
                result.codebook
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_strategy_display_auto() {
        assert_eq!(format!("{}", CodebookStrategy::Auto), "auto");
    }

    #[test]
    fn test_codebook_strategy_display_default() {
        assert_eq!(format!("{}", CodebookStrategy::Default), "default");
    }

    #[test]
    fn test_codebook_strategy_display_aotuv() {
        assert_eq!(format!("{}", CodebookStrategy::Aotuv), "aotuv");
    }

    #[test]
    fn test_codebook_strategy_default_is_auto() {
        assert_eq!(CodebookStrategy::default(), CodebookStrategy::Auto);
    }
}
