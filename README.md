# wemkit

A Rust toolkit for converting between WEM (Wwise Encoded Media) and standard audio formats.

## Features

- **Decode WEM → OGG/WAV**: Convert Wwise WEM files to standard audio formats
  - Automatic codebook detection (packed vs aoTuV)
  - Built-in validation of output files
  - Output to OGG or WAV (auto-detected from file extension)
- **Encode Audio → WEM**: Convert various audio formats to WEM
  - Supports WAV, OGG, MP3, FLAC, AAC input
  - Configurable Vorbis quality (High/Medium/Low)
  - Sample rate conversion with high-quality resampling
  - Channel conversion with proper surround downmix matrices (ITU-R BS.775)
  - Volume adjustment

## Installation

### From Source

```bash
git clone https://github.com/coconutbird/wemkit.git
cd wemkit
cargo build --release
```

The binary will be at `target/release/wemkit` (or `wemkit.exe` on Windows).

### Requirements

- **Decoding**: No external dependencies required
- **Encoding**: Requires [Audiokinetic Wwise](https://www.audiokinetic.com/products/wwise/) to be installed
  - Set `WWISEROOT` environment variable, or
  - Pass `--wwise-console` path directly

## Usage

### Decode WEM to OGG/WAV

```bash
# Decode to OGG (default)
wemkit decode -i input.wem -o output.ogg

# Decode to WAV (auto-detected from extension)
wemkit decode -i input.wem -o output.wav

# With verbose output
wemkit decode -i input.wem -o output.ogg -v

# Force specific codebook
wemkit decode -i input.wem -o output.ogg -c aotuv
```

### Encode Audio to WEM

```bash
# Basic usage (requires Wwise)
wemkit encode -i input.wav -o output.wem

# With quality setting
wemkit encode -i input.mp3 -o output.wem -q medium

# Convert to mono at 44.1kHz
wemkit encode -i input.ogg -o output.wem --channels 1 --sample-rate 44100

# Adjust volume (1.5 = 150%)
wemkit encode -i input.flac -o output.wem --volume 1.5
```

### Options

#### Decode Options

| Option              | Description                                     |
| ------------------- | ----------------------------------------------- |
| `-i, --input`       | Input WEM file                                  |
| `-o, --output`      | Output file (OGG or WAV, detected by extension) |
| `-c, --codebook`    | Codebook strategy: `auto`, `default`, `aotuv`   |
| `--skip-validation` | Skip output file validation                     |
| `-v, --verbose`     | Verbose output                                  |

#### Encode Options

| Option            | Description                                 |
| ----------------- | ------------------------------------------- |
| `-i, --input`     | Input audio file (WAV, OGG, MP3, FLAC, AAC) |
| `-o, --output`    | Output WEM file                             |
| `-q, --quality`   | Vorbis quality: `high`, `medium`, `low`     |
| `--sample-rate`   | Target sample rate in Hz                    |
| `--channels`      | Target channel count (1=mono, 2=stereo)     |
| `--volume`        | Volume multiplier (e.g., 0.5, 1.5)          |
| `--wwise-console` | Path to WwiseConsole.exe                    |
| `-v, --verbose`   | Verbose output                              |

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
wemkit = { git = "https://github.com/coconutbird/wemkit.git" }
```

```rust
use wemkit::{decode, encode, CodebookStrategy, EncodeConfig, VorbisQuality};
use std::path::PathBuf;

// Decode WEM to OGG
decode(
    &PathBuf::from("input.wem"),
    &PathBuf::from("output.ogg"),
    CodebookStrategy::Auto,
    false, // skip_validation
    true,  // verbose
)?;

// Decode WEM to WAV (format detected from extension)
decode(
    &PathBuf::from("input.wem"),
    &PathBuf::from("output.wav"),
    CodebookStrategy::Auto,
    false,
    false,
)?;

// Encode audio to WEM
let config = EncodeConfig {
    input: PathBuf::from("input.wav"),
    output: PathBuf::from("output.wem"),
    wwise_console: None, // Uses WWISEROOT env var
    quality: VorbisQuality::High,
    sample_rate: Some(48000),
    channels: Some(2),
    volume: None,
    verbose: true,
};
encode(config)?;
```

## Surround Sound Support

When downmixing surround audio to stereo, wemkit uses industry-standard ITU-R BS.775 coefficients:

| Source | Target | Method                           |
| ------ | ------ | -------------------------------- |
| 5.1    | Stereo | L = FL + C×0.707 + SL×0.707      |
| 7.1    | Stereo | Includes back channels at 0.5    |
| 7.1    | 5.1    | Back channels fold into surround |
| Quad   | Stereo | Rear attenuated by 3dB           |

LFE channel is excluded from stereo/mono downmix per industry standard.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [ww2ogg](https://github.com/hcs64/ww2ogg) - WEM to OGG conversion logic
- [Audiokinetic Wwise](https://www.audiokinetic.com/) - WEM encoding
