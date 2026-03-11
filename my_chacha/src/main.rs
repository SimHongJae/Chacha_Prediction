use rand::Rng;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

struct ChaCha {
    state: [u32; 16],
    rounds: u32,
}

impl ChaCha {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32, rounds: u32) -> Self {
        assert!(rounds % 2 == 0, "Rounds must be even");

        let mut s = [0u32; 16];

        // Constants (expand 32-byte k)
        s[0] = 0x61707865;
        s[1] = 0x3320646e;
        s[2] = 0x79622d32;
        s[3] = 0x6b206574;

        // Key (32 bytes -> 8 u32s)
        for i in 0..8 {
            let start = i * 4;
            s[4 + i] = u32::from_le_bytes(key[start..start + 4].try_into().unwrap());
        }

        // Counter (32-bit)
        s[12] = counter;

        // Nonce (12 bytes -> 3 u32s)
        s[13] = u32::from_le_bytes(nonce[0..4].try_into().unwrap());
        s[14] = u32::from_le_bytes(nonce[4..8].try_into().unwrap());
        s[15] = u32::from_le_bytes(nonce[8..12].try_into().unwrap());

        ChaCha { state: s, rounds }
    }

    fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(16);
        s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(12);
        s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(8);
        s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(7);
    }

    pub fn next_block(&mut self) -> [u8; 64] {
        let mut working = self.state;

        for _ in 0..self.rounds / 2 {
            Self::quarter_round(&mut working, 0, 4, 8, 12);
            Self::quarter_round(&mut working, 1, 5, 9, 13);
            Self::quarter_round(&mut working, 2, 6, 10, 14);
            Self::quarter_round(&mut working, 3, 7, 11, 15);

            Self::quarter_round(&mut working, 0, 5, 10, 15);
            Self::quarter_round(&mut working, 1, 6, 11, 12);
            Self::quarter_round(&mut working, 2, 7, 8, 13);
            Self::quarter_round(&mut working, 3, 4, 9, 14);
        }

        let mut output_bytes = [0u8; 64];
        for i in 0..16 {
            let word = working[i].wrapping_add(self.state[i]);
            output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        // Increment counter
        self.state[12] = self.state[12].wrapping_add(1);

        output_bytes
    }
}

fn parse_hex_key(s: &str) -> [u8; 32] {
    let s = s.trim();
    assert!(s.len() == 64, "Key must be 64 hex chars (32 bytes), got {}", s.len());
    let mut key = [0u8; 32];
    for i in 0..32 {
        key[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .expect("Invalid hex character in key");
    }
    key
}

fn parse_hex_nonce(s: &str) -> [u8; 12] {
    let s = s.trim();
    assert!(s.len() == 24, "Nonce must be 24 hex chars (12 bytes), got {}", s.len());
    let mut nonce = [0u8; 12];
    for i in 0..12 {
        nonce[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .expect("Invalid hex character in nonce");
    }
    nonce
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Usage:
///   my_chacha <rounds> <num_blocks> [key_hex] [nonce_hex]
///
/// Generates a single continuous ChaCha keystream with a fixed key.
/// Output: flat raw bytes (no length headers) -> ../dataset/chacha{rounds}_stream.bin
///
/// Arguments:
///   rounds      Number of ChaCha rounds (must be even, e.g. 4)
///   num_blocks  Number of 64-byte blocks to generate
///   key_hex     (optional) 64-char hex string = 32-byte key
///               If omitted, a random key is generated and printed
///   nonce_hex   (optional) 24-char hex string = 12-byte nonce
///               If omitted, a random nonce is generated and printed
///
/// Example:
///   my_chacha 4 1000000
///   my_chacha 4 1000000 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 000000000000000000000000
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        eprintln!("Usage: {} <rounds> <num_blocks> [key_hex] [nonce_hex]", args[0]);
        eprintln!("  rounds     : ChaCha rounds (even number, e.g. 4)");
        eprintln!("  num_blocks : number of 64-byte blocks to generate");
        eprintln!("  key_hex    : 64-char hex key (optional, random if omitted)");
        eprintln!("  nonce_hex  : 24-char hex nonce (optional, random if omitted)");
        eprintln!();
        eprintln!("Example: {} 4 1000000", args[0]);
        std::process::exit(1);
    }

    let rounds: u32 = args[1].parse().expect("Invalid rounds");
    let num_blocks: usize = args[2].parse().expect("Invalid num_blocks");
    assert!(rounds % 2 == 0, "Rounds must be even");

    let mut rng = rand::thread_rng();

    let key: [u8; 32] = if args.len() >= 4 {
        parse_hex_key(&args[3])
    } else {
        let mut k = [0u8; 32];
        rng.fill(&mut k);
        k
    };

    let nonce: [u8; 12] = if args.len() >= 5 {
        parse_hex_nonce(&args[4])
    } else {
        let mut n = [0u8; 12];
        rng.fill(&mut n);
        n
    };

    eprintln!("=== ChaCha{} Fixed-Key Stream Generator ===", rounds);
    eprintln!("Key  : {}", bytes_to_hex(&key));
    eprintln!("Nonce: {}", bytes_to_hex(&nonce));
    eprintln!("Blocks: {} ({} bytes = {:.1} MB)", num_blocks, num_blocks * 64,
              (num_blocks * 64) as f64 / (1024.0 * 1024.0));

    let dataset_dir = "../dataset";
    fs::create_dir_all(dataset_dir).expect("Failed to create dataset directory");

    let filename = format!("{}/chacha{}_stream.bin", dataset_dir, rounds);
    let file = File::create(&filename).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut chacha = ChaCha::new(key, nonce, 0, rounds);

    let report_interval = (num_blocks / 10).max(1);
    for block_idx in 0..num_blocks {
        let block = chacha.next_block();
        writer.write_all(&block).unwrap();

        if (block_idx + 1) % report_interval == 0 {
            eprintln!(
                "Progress: {}/{} blocks ({:.1}%)",
                block_idx + 1,
                num_blocks,
                (block_idx + 1) as f64 / num_blocks as f64 * 100.0,
            );
        }
    }

    writer.flush().unwrap();

    let total_bytes = num_blocks * 64;
    eprintln!("Done! {} created ({} bytes = {:.1} MB)",
              filename, total_bytes, total_bytes as f64 / (1024.0 * 1024.0));
    eprintln!();
    eprintln!("To reproduce this stream, run:");
    eprintln!("  {} {} {} {} {}", args[0], rounds, num_blocks,
              bytes_to_hex(&key), bytes_to_hex(&nonce));
}
