//! Audio to WEM encoding functionality

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

use crate::audio::convert_to_wav;

/// Vorbis quality level for WEM encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VorbisQuality {
    /// High quality (larger file size)
    #[default]
    High,
    /// Medium quality (balanced)
    Medium,
    /// Low quality (smaller file size)
    Low,
}

impl std::fmt::Display for VorbisQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VorbisQuality::High => write!(f, "Vorbis Quality High"),
            VorbisQuality::Medium => write!(f, "Vorbis Quality Medium"),
            VorbisQuality::Low => write!(f, "Vorbis Quality Low"),
        }
    }
}

/// Configuration for encoding audio to WEM
pub struct EncodeConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub wwise_console: Option<PathBuf>,
    pub quality: VorbisQuality,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub volume: Option<f32>,
    pub verbose: bool,
}

/// Create a Wwise project in the given directory
fn create_wwise_project(
    wwise_console: &Path,
    project_dir: &Path,
    verbose: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let project_file = project_dir.join("wemkit_conversion.wproj");

    if verbose {
        println!("  Creating Wwise project...");
    }

    let output = Command::new(wwise_console)
        .arg("create-new-project")
        .arg(&project_file)
        .arg("--quiet")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "Failed to create Wwise project:\nstdout: {}\nstderr: {}",
            stdout.trim(),
            stderr.trim()
        )
        .into());
    }

    if verbose {
        println!("  Created project at: {}", project_file.display());
    }

    Ok(project_file)
}

/// Find WwiseConsole.exe by searching within WWISEROOT directory
pub fn find_wwise_console() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let wwise_root = std::env::var("WWISEROOT").map_err(|_| {
        "WWISEROOT environment variable not set. Please install Wwise or specify --wwise-console path."
    })?;

    let root_path = PathBuf::from(&wwise_root);
    if !root_path.exists() {
        return Err(format!("WWISEROOT path does not exist: {}", wwise_root).into());
    }

    fn find_exe(dir: &Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() && path.file_name().is_some_and(|n| n == "WwiseConsole.exe") {
                return Some(path);
            }
            if path.is_dir()
                && let Some(found) = find_exe(&path)
            {
                return Some(found);
            }
        }
        None
    }

    find_exe(&root_path).ok_or_else(|| {
        format!(
            "WwiseConsole.exe not found within WWISEROOT: {}",
            wwise_root
        )
        .into()
    })
}

/// Generate wsources XML file for Wwise conversion
fn generate_wsources(
    audio_dir: &Path,
    audio_files: &[PathBuf],
    quality: VorbisQuality,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str(&format!(
        "<ExternalSourcesList SchemaVersion=\"1\" Root=\"{}\">\n",
        audio_dir.display()
    ));

    for file in audio_files {
        let filename = file
            .file_name()
            .ok_or("Invalid filename")?
            .to_string_lossy();
        xml.push_str(&format!(
            "\t<Source Path=\"{}\" Conversion=\"{}\"/>\n",
            filename, quality
        ));
    }

    xml.push_str("</ExternalSourcesList>\n");
    Ok(xml)
}

/// Encode an audio file to WEM format
pub fn encode(config: EncodeConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Converting {} -> {}",
        config.input.display(),
        config.output.display()
    );

    // Find or validate WwiseConsole
    let wwise_console = match config.wwise_console {
        Some(path) => {
            if !path.exists() {
                return Err(format!("WwiseConsole.exe not found at: {}", path.display()).into());
            }
            path
        }
        None => {
            if config.verbose {
                println!("  Auto-detecting WwiseConsole...");
            }
            find_wwise_console()?
        }
    };

    if config.verbose {
        println!("  Using WwiseConsole: {}", wwise_console.display());
    }

    // Create temp directory for processing (auto-cleaned on drop or OS cleanup)
    let temp_dir = TempDir::with_prefix("wemkit_")?;
    let temp_path = temp_dir.path();

    // Create Wwise project in temp directory (thread-safe, each encode gets its own)
    let project_file = create_wwise_project(&wwise_console, temp_path, config.verbose)?;

    // Check if input is already WAV
    let is_wav = config
        .input
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));

    // Wwise only accepts WAV input, so we need conversion for non-WAV files
    // Also convert if audio adjustments are requested
    let needs_conversion = !is_wav
        || config.sample_rate.is_some()
        || config.channels.is_some()
        || config.volume.is_some();

    let processed_input = if needs_conversion {
        let input_stem = config
            .input
            .file_stem()
            .ok_or("Invalid input filename")?
            .to_string_lossy();
        let wav_path = temp_path.join(format!("{}.wav", input_stem));

        println!("  Converting to WAV...");
        convert_to_wav(
            &config.input,
            &wav_path,
            config.sample_rate,
            config.channels,
            config.volume,
            config.verbose,
        )?;

        wav_path
    } else {
        // Input is WAV and no adjustments needed - copy to temp dir
        let input_filename = config.input.file_name().ok_or("Invalid input filename")?;
        let temp_input = temp_path.join(input_filename);
        std::fs::copy(&config.input, &temp_input)?;
        temp_input
    };

    // Generate wsources file
    let wsources_path = temp_path.join("sources.wsources");
    let wsources_content = generate_wsources(
        temp_path,
        std::slice::from_ref(&processed_input),
        config.quality,
    )?;
    std::fs::write(&wsources_path, &wsources_content)?;

    if config.verbose {
        println!("  Generated wsources:\n{}", wsources_content);
    }

    // Create output directory
    let wem_output_dir = temp_path.join("output");
    std::fs::create_dir_all(&wem_output_dir)?;

    // Run Wwise conversion
    println!("  Converting with Wwise (quality: {})...", config.quality);
    let cmd_output = Command::new(&wwise_console)
        .arg("convert-external-source")
        .arg(&project_file)
        .arg("--source-file")
        .arg(&wsources_path)
        .arg("--output")
        .arg(&wem_output_dir)
        .arg("--quiet")
        .output()?;

    if config.verbose {
        if !cmd_output.stdout.is_empty() {
            println!("  stdout: {}", String::from_utf8_lossy(&cmd_output.stdout));
        }
        if !cmd_output.stderr.is_empty() {
            eprintln!("  stderr: {}", String::from_utf8_lossy(&cmd_output.stderr));
        }
    }

    if !cmd_output.status.success() {
        return Err(format!(
            "WwiseConsole failed with exit code: {:?}\nstdout: {}\nstderr: {}",
            cmd_output.status.code(),
            String::from_utf8_lossy(&cmd_output.stdout).trim(),
            String::from_utf8_lossy(&cmd_output.stderr).trim()
        )
        .into());
    }

    // Find and move the output WEM file
    let windows_output = wem_output_dir.join("Windows");
    let wem_files: Vec<_> = std::fs::read_dir(&windows_output)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wem"))
        .collect();

    if wem_files.is_empty() {
        return Err("No WEM file was generated".into());
    }

    let generated_wem = &wem_files[0].path();

    // Ensure output directory exists
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::copy(generated_wem, &config.output)?;

    let file_size = std::fs::metadata(&config.output)?.len();
    println!(
        "✓ Conversion complete! ({} bytes, quality: {})",
        file_size, config.quality
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wsources_single_file() {
        let audio_dir = PathBuf::from("C:\\temp\\audio");
        let audio_files = vec![PathBuf::from("C:\\temp\\audio\\test.wav")];

        let result = generate_wsources(&audio_dir, &audio_files, VorbisQuality::High).unwrap();

        assert!(result.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(result.contains("SchemaVersion=\"1\""));
        assert!(result.contains("Root=\"C:\\temp\\audio\""));
        assert!(result.contains("Path=\"test.wav\""));
        assert!(result.contains("Conversion=\"Vorbis Quality High\""));
    }

    #[test]
    fn test_generate_wsources_multiple_files() {
        let audio_dir = PathBuf::from("/audio");
        let audio_files = vec![
            PathBuf::from("/audio/file1.wav"),
            PathBuf::from("/audio/file2.wav"),
            PathBuf::from("/audio/file3.wav"),
        ];

        let result = generate_wsources(&audio_dir, &audio_files, VorbisQuality::Medium).unwrap();

        assert!(result.contains("Path=\"file1.wav\""));
        assert!(result.contains("Path=\"file2.wav\""));
        assert!(result.contains("Path=\"file3.wav\""));
        assert!(result.contains("Conversion=\"Vorbis Quality Medium\""));
    }

    #[test]
    fn test_vorbis_quality_display_high() {
        assert_eq!(format!("{}", VorbisQuality::High), "Vorbis Quality High");
    }

    #[test]
    fn test_vorbis_quality_display_medium() {
        assert_eq!(
            format!("{}", VorbisQuality::Medium),
            "Vorbis Quality Medium"
        );
    }

    #[test]
    fn test_vorbis_quality_display_low() {
        assert_eq!(format!("{}", VorbisQuality::Low), "Vorbis Quality Low");
    }

    #[test]
    fn test_vorbis_quality_default_is_high() {
        assert_eq!(VorbisQuality::default(), VorbisQuality::High);
    }

    #[test]
    fn test_generate_wsources_empty_files() {
        let audio_dir = PathBuf::from("/audio");
        let audio_files: Vec<PathBuf> = vec![];

        let result = generate_wsources(&audio_dir, &audio_files, VorbisQuality::Low).unwrap();

        assert!(result.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(result.contains("</ExternalSourcesList>"));
        // Should not contain any Source elements
        assert!(!result.contains("<Source"));
    }
}
