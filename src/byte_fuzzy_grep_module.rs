// File: src/main.rs
// Fuzzy grep tool using byte-level Levenshtein distance with parallel processing

/*

    let pattern_input = &args[1];
    let threshold: usize = args[2].parse()?;
    let search_dir = Path::new(&args[3]);

# Search for text pattern
./fuzzy_grep "hello" 2 /path/to/search

# Search for hex bytes
./fuzzy_grep 0x48656c6c6f 2 /path/to/search

# Search with spaces in hex
./fuzzy_grep "0x48 65 6c 6c 6f" 2 /path/to/search

TODO:
regular expression...
fuzzy
bytes...

grep options
- only text files
- only non-hidden files
- case sensitive
- levenshtein
-

idea: common line max length,
zip through file until max line length and sent to parallel processing
if shorter (most cases) just send
if longer: manage overlapp-chunk issues...
not-duplicate matches...
but not divided-matches either.

(look up how zip-grep is faster?)
what are standard parts of grep to include?
book on command-line rust...


https://ast-grep.github.io/blog/optimize-ast-grep.html

- link-page return
- pagination
- Q&A interface
- modular...output that FF needs

some kind of help screen... commands?

rust optimized grep

why is ripgrep faster?
https://www.reddit.com/r/rust/comments/sr02aj/what_makes_ripgrep_so_fast/

what are trade-offs?
- simpler...
- modular...
- parallel?

How to add fuzzy-byte-grep?
how to input the parameters?
Q&A?
maybe use Q&A for byte_grep mode
e.g. specifying:
- fuzzy or normal
- nest depth limit (or full)
- start path
- nonhidden files
- text files only or all files
- encoding type if text (Q: how to set up handling encoding options?)
- boolean or text
- thread count
- nested directory depth limit would be nice
- regex pattern option
- boolean input option
- boolean array option
- character/string input option


option for output as:
1. short list
2. details json


https://github.com/lineality/fuzzy_byte_grep_module_rust
use enums to make sure the input is type specific

idea: using levenshtein byte search for characters of bytes, selecting encoding for text input

the cli is a good test interface, but for use as a module
there should be a more systematic interface, e.g. to be called within another function.
e.g. specifying:
- start path
- text files only or all files
- encoding type if text
- boolean or text
- thread count
- nested directory depth limit would be nice
- regex pattern option
- boolean input option
- boolean array option
- character/string input option

*/

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for the fuzzy search operation
#[derive(Clone)]
pub struct SearchConfig {
    /// The pattern to search for (as bytes)
    pattern_bytes: Vec<u8>,
    /// Maximum Levenshtein distance threshold for matches
    threshold: usize,
    /// Whether to search only text files (faster) or all files
    text_files_only: bool,
    /// Number of threads to use for parallel processing
    num_threads: usize,
    /// Size of chunks when reading large files (in bytes)
    chunk_size: usize,
    /// Size of overlap between chunks to avoid missing matches at boundaries
    overlap_size: usize,
    /// Whether to interpret input as UTF-8 string or raw bytes
    use_utf8: bool,
}

/// Represents a match found during the search
#[derive(Debug, Clone)]
pub struct ByteGrepMatch {
    /// Absolute path to the file containing the match
    file_path: String,
    /// Line number (if applicable, 0 for binary files)
    line_number: usize,
    /// Byte offset in the file where the match starts
    byte_offset: usize,
    /// The matched bytes
    matched_bytes: Vec<u8>,
    /// The Levenshtein distance of the match
    distance: usize,
    /// Context around the match (for display purposes)
    context: String,
}

/// Calculate byte-level Levenshtein distance between two byte sequences
///
/// # Arguments
/// * `a` - First byte sequence
/// * `b` - Second byte sequence
///
/// # Returns
/// The Levenshtein distance between the two sequences
pub fn byte_levenshtein_distance(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();

    // Create a matrix to store distances
    let mut matrix = vec![vec![0; len_b + 1]; len_a + 1];

    // Initialize first row and column
    for i in 0..=len_a {
        matrix[i][0] = i;
    }
    for j in 0..=len_b {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1,      // deletion
                    matrix[i][j - 1] + 1        // insertion
                ),
                matrix[i - 1][j - 1] + cost    // substitution
            );
        }
    }

    matrix[len_a][len_b]
}

/// Find fuzzy matches in a byte buffer using sliding window approach
///
/// # Arguments
/// * `buffer` - The byte buffer to search in
/// * `pattern` - The pattern bytes to search for
/// * `threshold` - Maximum Levenshtein distance for matches
/// * `start_offset` - Byte offset of this buffer in the original file
///
/// # Returns
/// Vector of tuples containing (byte_offset, matched_bytes, distance)
pub fn fuzzy_match_bytes(
    buffer: &[u8],
    pattern: &[u8],
    threshold: usize,
    start_offset: usize
) -> Vec<(usize, Vec<u8>, usize)> {
    let mut matches = Vec::new();
    let pattern_len = pattern.len();

    if buffer.len() < pattern_len {
        return matches;
    }

    // Calculate min and max window sizes based on threshold
    let min_window = pattern_len.saturating_sub(threshold);
    let max_window = pattern_len + threshold;

    // Sliding window approach
    for i in 0..buffer.len() {
        for window_size in min_window..=max_window {
            if i + window_size > buffer.len() {
                break;
            }

            let window = &buffer[i..i + window_size];
            let distance = byte_levenshtein_distance(pattern, window);

            if distance <= threshold {
                matches.push((
                    start_offset + i,
                    window.to_vec(),
                    distance
                ));
            }
        }
    }

    // Remove duplicate/overlapping matches, keeping the best ones
    deduplicate_matches(&mut matches);

    matches
}

/// Remove duplicate or overlapping matches, keeping those with lowest distance
pub fn deduplicate_matches(matches: &mut Vec<(usize, Vec<u8>, usize)>) {
    if matches.is_empty() {
        return;
    }

    // Sort by position and distance
    matches.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.2.cmp(&b.2))
    });

    let mut filtered = Vec::new();
    let mut last_end = 0;

    for (offset, bytes, distance) in matches.iter() {
        if *offset >= last_end {
            filtered.push((*offset, bytes.clone(), *distance));
            last_end = offset + bytes.len();
        }
    }

    *matches = filtered;
}

/// Check if a file has a known plain text extension
pub fn is_text_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            matches!(ext_str.to_lowercase().as_str(),
                "txt" | "md" | "rs" | "py" | "js" | "html" | "css" | "json" |
                "xml" | "csv" | "log" | "cfg" | "conf" | "ini" | "yaml" | "yml" |
                "toml" | "sh" | "bat" | "c" | "cpp" | "h" | "hpp" | "java" | "go" |
                "php" | "rb" | "pl" | "lua" | "sql" | "r" | "scala" | "kt" | "swift" |
                "asm" | "s" | "makefile" | "mk" | "dockerfile" | "tex" | "bib"
            )
        } else {
            false
        }
    } else {
        false
    }
}

/// Search for pattern in a single file using chunked reading
pub fn search_in_file(
    file_path: &Path,
    config: &SearchConfig
) -> Result<Vec<ByteGrepMatch>, Box<dyn std::error::Error>> {
    let mut matches = Vec::new();
    let absolute_path = fs::canonicalize(file_path)?;
    let path_str = absolute_path.to_str()
        .ok_or("Invalid path encoding")?
        .to_string();

    let file = File::open(&absolute_path)?;
    let file_size = file.metadata()?.len() as usize;

    // For small files, read entirely into memory
    if file_size <= config.chunk_size {
        let mut buffer = Vec::new();
        let mut reader = BufReader::new(file);
        reader.read_to_end(&mut buffer)?;

        let found = fuzzy_match_bytes(&buffer, &config.pattern_bytes, config.threshold, 0);

        for (offset, matched_bytes, distance) in found {
            let context = extract_context(&buffer, offset, matched_bytes.len());
            matches.push(ByteGrepMatch {
                file_path: path_str.clone(),
                line_number: calculate_line_number(&buffer, offset),
                byte_offset: offset,
                matched_bytes,
                distance,
                context,
            });
        }
    } else {
        // For large files, use chunked reading with overlap
        let mut reader = BufReader::new(file);
        let mut current_offset: usize = 0;
        let mut previous_overlap = Vec::new();

        loop {
            let mut chunk = vec![0u8; config.chunk_size];
            let bytes_read = reader.read(&mut chunk)?;

            if bytes_read == 0 {
                break;
            }

            chunk.truncate(bytes_read);

            // Combine with previous overlap
            let mut search_buffer = previous_overlap.clone();
            search_buffer.extend_from_slice(&chunk);

            // Search in the combined buffer
            let found = fuzzy_match_bytes(
                &search_buffer,
                &config.pattern_bytes,
                config.threshold,
                current_offset.saturating_sub(previous_overlap.len())
            );

            for (offset, matched_bytes, distance) in found {
                // Skip matches in the overlap region that were already found
                if offset >= current_offset || previous_overlap.is_empty() {
                    let context = extract_context(&search_buffer,
                        offset - current_offset.saturating_sub(previous_overlap.len()),
                        matched_bytes.len());
                    matches.push(ByteGrepMatch {
                        file_path: path_str.clone(),
                        line_number: 0, // Line numbers are expensive for large binary files
                        byte_offset: offset,
                        matched_bytes,
                        distance,
                        context,
                    });
                }
            }

            // Prepare overlap for next iteration
            if bytes_read == config.chunk_size {
                let overlap_start = bytes_read.saturating_sub(config.overlap_size);
                previous_overlap = chunk[overlap_start..].to_vec();
            } else {
                previous_overlap.clear();
            }

            current_offset += bytes_read;
        }
    }

    Ok(matches)
}

/// Calculate line number for a given byte offset
fn calculate_line_number(buffer: &[u8], offset: usize) -> usize {
    let mut line_num = 1;
    for i in 0..std::cmp::min(offset, buffer.len()) {
        if buffer[i] == b'\n' {
            line_num += 1;
        }
    }
    line_num
}

/// Extract context around a match for display purposes
fn extract_context(buffer: &[u8], offset: usize, match_len: usize) -> String {
    let context_before = 30;
    let context_after = 30;

    let start = offset.saturating_sub(context_before);
    let end = std::cmp::min(buffer.len(), offset + match_len + context_after);

    let context_bytes = &buffer[start..end];

    // Try to convert to UTF-8 for display, fallback to lossy conversion
    String::from_utf8_lossy(context_bytes).into_owned()
}

/// Process files in parallel using thread pool
pub fn parallel_search(
    files: Vec<PathBuf>,
    config: SearchConfig
) -> Vec<ByteGrepMatch> {
    let shared_files = Arc::new(Mutex::new(files));
    let shared_results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for _ in 0..config.num_threads {
        let files_clone = Arc::clone(&shared_files);
        let results_clone = Arc::clone(&shared_results);
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            loop {
                // Get next file to process
                let file_path = {
                    let mut files = files_clone.lock()
                        .expect("Failed to lock files mutex");
                    files.pop()
                };

                match file_path {
                    Some(path) => {
                        // Process the file
                        match search_in_file(&path, &config_clone) {
                            Ok(matches) => {
                                if !matches.is_empty() {
                                    let mut results = results_clone.lock()
                                        .expect("Failed to lock results mutex");
                                    results.extend(matches);
                                }
                            }
                            Err(e) => {
                                eprintln!("Error processing {:?}: {}", path, e);
                            }
                        }
                    }
                    None => break, // No more files to process
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    Arc::try_unwrap(shared_results)
        .expect("Failed to unwrap results")
        .into_inner()
        .expect("Failed to extract results")
}

/// Recursively find all files in a directory
pub fn find_files(dir: &Path, text_only: bool) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recursively search subdirectories
            files.extend(find_files(&path, text_only)?);
        } else if path.is_file() {
            if !text_only || is_text_file(&path) {
                files.push(path);
            }
        }
    }

    Ok(files)
}

/// Convert search results to JSON format
pub fn results_to_json(matches: &[ByteGrepMatch]) -> String {
    let mut json = String::from("{\n");
    json.push_str("  \"search_results\": [\n");

    for (i, m) in matches.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"file_path\": \"{}\",\n",
            m.file_path.replace('\\', "\\\\").replace('"', "\\\"")));
        json.push_str(&format!("      \"line_number\": {},\n", m.line_number));
        json.push_str(&format!("      \"byte_offset\": {},\n", m.byte_offset));
        json.push_str(&format!("      \"distance\": {},\n", m.distance));

        // Convert matched bytes to hex string for JSON
        let hex_string: String = m.matched_bytes.iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        json.push_str(&format!("      \"matched_bytes_hex\": \"{}\",\n", hex_string));

        // Try to convert to UTF-8 string if possible
        if let Ok(utf8_str) = String::from_utf8(m.matched_bytes.clone()) {
            json.push_str(&format!("      \"matched_text\": \"{}\",\n",
                utf8_str.replace('\\', "\\\\").replace('"', "\\\"")));
        }

        json.push_str(&format!("      \"context\": \"{}\"\n",
            m.context.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")));

        if i < matches.len() - 1 {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }

    json.push_str("  ],\n");
    json.push_str(&format!("  \"total_matches\": {}\n", matches.len()));
    json.push_str("}\n");

    json
}

/// Pretty print search results to terminal
pub fn pretty_print_results(matches: &[ByteGrepMatch]) {
    println!("\n{}", "=".repeat(80));
    println!("FUZZY SEARCH RESULTS");
    println!("{}", "=".repeat(80));
    println!("Total matches found: {}\n", matches.len());

    // Group matches by file
    let mut by_file: HashMap<String, Vec<&ByteGrepMatch>> = HashMap::new();
    for m in matches {
        by_file.entry(m.file_path.clone())
            .or_insert_with(Vec::new)
            .push(m);
    }

    for (file_path, file_matches) in by_file.iter() {
        println!("\n{}", "-".repeat(80));
        println!("File: {}", file_path);
        println!("Matches: {}", file_matches.len());
        println!("{}", "-".repeat(80));

        for m in file_matches {
            println!("\n  Distance: {}", m.distance);
            if m.line_number > 0 {
                println!("  Line: {}", m.line_number);
            }
            println!("  Byte offset: {}", m.byte_offset);

            // Try to display as UTF-8 text
            if let Ok(text) = String::from_utf8(m.matched_bytes.clone()) {
                println!("  Matched text: \"{}\"", text);
            } else {
                // Display as hex for binary data
                let hex: String = m.matched_bytes.iter()
                    .take(20)
                    .map(|b| format!("{:02x} ", b))
                    .collect();
                println!("  Matched bytes (hex): {}{}",
                    hex,
                    if m.matched_bytes.len() > 20 { "..." } else { "" });
            }

            println!("  Context: {}",
                m.context.replace('\n', "\\n")
                    .chars()
                    .take(100)
                    .collect::<String>());
        }
    }

    println!("\n{}", "=".repeat(80));
}

/// Parse hex string into bytes
pub fn parse_hex_string(hex: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let hex = hex.replace(' ', "").replace("0x", "");

    if hex.len() % 2 != 0 {
        return Err("Hex string must have even length".into());
    }

    let mut bytes = Vec::new();
    for i in (0..hex.len()).step_by(2) {
        let byte_str = &hex[i..i+2];
        let byte = u8::from_str_radix(byte_str, 16)?;
        bytes.push(byte);
    }

    Ok(bytes)
}

/// Main entry point for the fuzzy grep tool
pub fn byte_fuzzy_grep_cli() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <pattern> <threshold> <directory>", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  pattern    - Text pattern or hex bytes (prefix with 0x for hex)");
        eprintln!("  threshold  - Maximum Levenshtein distance for matches");
        eprintln!("  directory  - Directory to search recursively");
        eprintln!("\nExample:");
        eprintln!("  {} \"hello\" 2 /path/to/search", args[0]);
        eprintln!("  {} 0x48656c6c6f 2 /path/to/search", args[0]);
        return Ok(());
    }

    let pattern_input = &args[1];
    let threshold: usize = args[2].parse()?;
    let search_dir = Path::new(&args[3]);

    if !search_dir.exists() {
        return Err(format!("Directory does not exist: {:?}", search_dir).into());
    }

    // Parse pattern (check if hex or text)
    let pattern_bytes = if pattern_input.starts_with("0x") {
        parse_hex_string(pattern_input)?
    } else {
        pattern_input.as_bytes().to_vec()
    };

    println!("Pattern bytes ({} bytes): {:?}",
        pattern_bytes.len(),
        pattern_bytes.iter().take(20).collect::<Vec<_>>());

    // Ask user about search scope
    println!("\nSearch options:");
    println!("1. Search only text files (faster)");
    println!("2. Search all files (includes binary)");
    print!("Enter choice (1 or 2): ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;
    let text_files_only = choice.trim() == "1";

    // Configure search
    let num_cpus = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let num_threads = std::cmp::max(1, num_cpus - 1);

    let config = SearchConfig {
        pattern_bytes,
        threshold,
        text_files_only,
        num_threads,
        chunk_size: 1024 * 1024,  // 1MB chunks
        overlap_size: 1024,        // 1KB overlap
        use_utf8: !pattern_input.starts_with("0x"),
    };

    println!("\nSearching with {} threads...", num_threads);
    println!("Threshold: {} edits", threshold);
    println!("Mode: {}", if text_files_only { "Text files only" } else { "All files" });

    // Find all files to search
    let files = find_files(search_dir, text_files_only)?;
    println!("Found {} files to search", files.len());

    if files.is_empty() {
        println!("No files found to search.");
        return Ok(());
    }

    // Perform parallel search
    let start_time = SystemTime::now();
    let matches = parallel_search(files, config);
    let duration = start_time.elapsed()?;

    println!("\nSearch completed in {:.2} seconds", duration.as_secs_f64());

    if matches.is_empty() {
        println!("No matches found.");
        return Ok(());
    }

    // Pretty print results
    pretty_print_results(&matches);

    // Generate JSON output
    let json = results_to_json(&matches);

    // Save JSON to file with timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs();
    let pattern_safe = pattern_input
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>();
    let json_filename = format!("{}_searchresults_{}.json",
        if pattern_safe.is_empty() { "fuzzy" } else { &pattern_safe },
        timestamp);

    let mut json_file = File::create(&json_filename)?;
    json_file.write_all(json.as_bytes())?;

    println!("\nResults saved to: {}", json_filename);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_levenshtein_distance() {
        // Test identical sequences
        assert_eq!(byte_levenshtein_distance(b"hello", b"hello"), 0);

        // Test single character difference
        assert_eq!(byte_levenshtein_distance(b"hello", b"hallo"), 1);

        // Test completely different sequences
        assert_eq!(byte_levenshtein_distance(b"abc", b"xyz"), 3);

        // Test different lengths
        assert_eq!(byte_levenshtein_distance(b"cat", b"catch"), 2);

        // Test empty sequences
        assert_eq!(byte_levenshtein_distance(b"", b"abc"), 3);
        assert_eq!(byte_levenshtein_distance(b"abc", b""), 3);
        assert_eq!(byte_levenshtein_distance(b"", b""), 0);
    }

    #[test]
    fn test_fuzzy_match_bytes() {
        let buffer = b"The quick brown fox jumps over the lazy dog";
        let pattern = b"brwn";
        let threshold = 1;

        let matches = fuzzy_match_bytes(buffer, pattern, threshold, 0);

        // Should find "brown" with distance 1
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|(_, bytes, dist)| {
            bytes == b"brown" && *dist == 1
        }));
    }

    #[test]
    fn test_is_text_file() {
        use std::path::PathBuf;

        assert!(is_text_file(&PathBuf::from("test.txt")));
        assert!(is_text_file(&PathBuf::from("code.rs")));
        assert!(is_text_file(&PathBuf::from("script.py")));
        assert!(!is_text_file(&PathBuf::from("image.jpg")));
        assert!(!is_text_file(&PathBuf::from("binary.exe")));
        assert!(!is_text_file(&PathBuf::from("no_extension")));
    }

    #[test]
    fn test_parse_hex_string() {
        // Test valid hex strings
        assert_eq!(parse_hex_string("48656c6c6f").unwrap(), b"Hello");
        assert_eq!(parse_hex_string("0x48656c6c6f").unwrap(), b"Hello");
        assert_eq!(parse_hex_string("48 65 6c 6c 6f").unwrap(), b"Hello");

        // Test invalid hex strings
        assert!(parse_hex_string("4865g").is_err()); // Invalid character
        assert!(parse_hex_string("486").is_err());   // Odd length
    }

    #[test]
    fn test_deduplicate_matches() {
        let mut matches = vec![
            (0, vec![1, 2, 3], 1),
            (2, vec![3, 4], 2),    // Overlaps with first
            (5, vec![5, 6], 1),    // No overlap
            (5, vec![5, 6, 7], 0), // Same position but better distance
        ];

        deduplicate_matches(&mut matches);

        // Should keep non-overlapping matches
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].0, 0);
        assert_eq!(matches[1].0, 5);
    }
}
