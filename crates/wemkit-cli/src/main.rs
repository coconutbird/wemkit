//! WEM/OGG audio conversion CLI

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use wemkit::{CodebookStrategy, EncodeConfig, VorbisQuality};

/// WEM/OGG audio conversion tool
#[derive(Parser)]
#[command(name = "wemkit")]
#[command(about = "Convert between WEM and OGG audio formats")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Decode WEM files to OGG format
    Decode {
        /// Input WEM file
        #[arg(short, long)]
        input: PathBuf,

        /// Output OGG file
        #[arg(short, long)]
        output: PathBuf,

        /// Codebook strategy for decoding
        #[arg(short = 'c', long, default_value = "auto")]
        codebook: CliCodebookStrategy,

        /// Skip validation of the output OGG file
        #[arg(long)]
        skip_validation: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Encode audio files to WEM format
    Encode {
        /// Input audio file (WAV, OGG, MP3, FLAC, AAC)
        #[arg(short, long)]
        input: PathBuf,

        /// Output WEM file
        #[arg(short, long)]
        output: PathBuf,

        /// Path to WwiseConsole.exe (uses WWISEROOT env var if not specified)
        #[arg(long, env = "WWISE_CONSOLE")]
        wwise_console: Option<PathBuf>,

        /// Vorbis quality level
        #[arg(short, long, default_value = "high")]
        quality: CliVorbisQuality,

        /// Audio sample rate in Hz (e.g., 44100, 48000)
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Number of audio channels (1 = mono, 2 = stereo)
        #[arg(long)]
        channels: Option<u8>,

        /// Volume adjustment multiplier (e.g., 1.5 for 150%, 0.5 for 50%)
        #[arg(long)]
        volume: Option<f32>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Codebook strategy for CLI (with ValueEnum derive)
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliCodebookStrategy {
    Auto,
    Default,
    Aotuv,
}

impl From<CliCodebookStrategy> for CodebookStrategy {
    fn from(cli: CliCodebookStrategy) -> Self {
        match cli {
            CliCodebookStrategy::Auto => CodebookStrategy::Auto,
            CliCodebookStrategy::Default => CodebookStrategy::Default,
            CliCodebookStrategy::Aotuv => CodebookStrategy::Aotuv,
        }
    }
}

/// Vorbis quality for CLI (with ValueEnum derive)
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliVorbisQuality {
    High,
    Medium,
    Low,
}

impl From<CliVorbisQuality> for VorbisQuality {
    fn from(cli: CliVorbisQuality) -> Self {
        match cli {
            CliVorbisQuality::High => VorbisQuality::High,
            CliVorbisQuality::Medium => VorbisQuality::Medium,
            CliVorbisQuality::Low => VorbisQuality::Low,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Decode {
            input,
            output,
            codebook,
            skip_validation,
            verbose,
        } => wemkit::decode(&input, &output, codebook.into(), skip_validation, verbose),
        Commands::Encode {
            input,
            output,
            wwise_console,
            quality,
            sample_rate,
            channels,
            volume,
            verbose,
        } => {
            let config = EncodeConfig {
                input,
                output,
                wwise_console,
                quality: quality.into(),
                sample_rate,
                channels,
                volume,
                verbose,
            };
            wemkit::encode(config)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
