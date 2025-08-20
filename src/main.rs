// src/main.rs

// import file fantstic module w/ these 2 lines
mod byte_fuzzy_grep_module;
use byte_fuzzy_grep_module::byte_fuzzy_grep_cli;

fn main() {
    
    // Let's call File Fantastic Go!!
    if let Err(e) = byte_fuzzy_grep_cli() {
        
        // Handle errors
        eprintln!("Error: {}", e);
        
        // exit code one means ok!
        std::process::exit(1);
    }
}

