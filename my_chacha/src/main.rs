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

        s[0] = 0x61707865;
        s[1] = 0x3320646e;
        s[2] = 0x79622d32;
        s[3] = 0x6b206574;

        for i in 0..8 {
            let start = i * 4;
            s[4 + i] = u32::from_le_bytes(key[start..start + 4].try_into().unwrap());
        }

        s[12] = counter;

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

        self.state[12] = self.state[12].wrapping_add(1);

        output_bytes
    }

    /// Compute a single block without advancing the counter.
    pub fn block_for(key: [u8; 32], nonce: [u8; 12], counter: u32, rounds: u32) -> [u8; 64] {
        let mut c = ChaCha::new(key, nonce, counter, rounds);
        c.next_block()
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

/// Mode: stream
///   my_chacha <rounds> stream <num_blocks> [key_hex] [nonce_hex]
///   Vary counter 0..num_blocks with fixed key+nonce.
///   Output: flat raw bytes -> ../dataset/chacha{rounds}_stream.bin
///   Record: [block_output: 64 bytes]  (counter = file_position / 64)
///
/// Mode: counter
///   my_chacha <rounds> counter <num_blocks> [key_hex] [nonce_hex]
///   Same as stream but saves (counter_le32 + block_output) pairs.
///   Output: ../dataset/chacha{rounds}_counter_sweep.bin
///   Record: [counter: 4 bytes LE] [block_output: 64 bytes] = 68 bytes
///
/// Mode: nonce
///   my_chacha <rounds> nonce <num_samples> [key_hex]
///   Vary nonce randomly with fixed key + counter=0.
///   Output: ../dataset/chacha{rounds}_nonce_sweep.bin
///   Record: [nonce: 12 bytes] [block_output: 64 bytes] = 76 bytes
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage:");
        eprintln!("  {} <rounds> stream  <num_blocks> [key_hex] [nonce_hex]", args[0]);
        eprintln!("  {} <rounds> counter <num_blocks> [key_hex] [nonce_hex]", args[0]);
        eprintln!("  {} <rounds> nonce   <num_samples> [key_hex]", args[0]);
        std::process::exit(1);
    }

    let rounds: u32 = args[1].parse().expect("Invalid rounds");
    assert!(rounds % 2 == 0, "Rounds must be even");
    let mode = args[2].as_str();
    let count: usize = args[3].parse().expect("Invalid count");

    let mut rng = rand::thread_rng();
    let dataset_dir = "../dataset";
    fs::create_dir_all(dataset_dir).expect("Failed to create dataset directory");

    match mode {
        "stream" | "counter" => {
            let key: [u8; 32] = if args.len() >= 5 {
                parse_hex_key(&args[4])
            } else {
                let mut k = [0u8; 32]; rng.fill(&mut k); k
            };
            let nonce: [u8; 12] = if args.len() >= 6 {
                parse_hex_nonce(&args[5])
            } else {
                let mut n = [0u8; 12]; rng.fill(&mut n); n
            };

            eprintln!("=== ChaCha{} {} sweep ===", rounds, mode);
            eprintln!("Key  : {}", bytes_to_hex(&key));
            eprintln!("Nonce: {}", bytes_to_hex(&nonce));
            eprintln!("Blocks: {}", count);

            let filename = if mode == "stream" {
                format!("{}/chacha{}_stream.bin", dataset_dir, rounds)
            } else {
                format!("{}/chacha{}_counter_sweep.bin", dataset_dir, rounds)
            };

            let file = File::create(&filename).expect("Failed to create output file");
            let mut writer = BufWriter::new(file);
            let mut chacha = ChaCha::new(key, nonce, 0, rounds);
            let report_interval = (count / 10).max(1);

            for block_idx in 0..count {
                let block = chacha.next_block();
                if mode == "counter" {
                    writer.write_all(&(block_idx as u32).to_le_bytes()).unwrap();
                }
                writer.write_all(&block).unwrap();

                if (block_idx + 1) % report_interval == 0 {
                    eprintln!("Progress: {}/{} ({:.0}%)", block_idx + 1, count,
                              (block_idx + 1) as f64 / count as f64 * 100.0);
                }
            }
            writer.flush().unwrap();
            eprintln!("Done! {}", filename);
            eprintln!("To reproduce: {} {} {} {} {} {}",
                      args[0], rounds, mode, count,
                      bytes_to_hex(&key), bytes_to_hex(&nonce));
        }

        "nonce" => {
            let key: [u8; 32] = if args.len() >= 5 {
                parse_hex_key(&args[4])
            } else {
                let mut k = [0u8; 32]; rng.fill(&mut k); k
            };

            eprintln!("=== ChaCha{} nonce sweep ===", rounds);
            eprintln!("Key    : {}", bytes_to_hex(&key));
            eprintln!("Counter: 0 (fixed)");
            eprintln!("Samples: {}", count);

            let filename = format!("{}/chacha{}_nonce_sweep.bin", dataset_dir, rounds);
            let file = File::create(&filename).expect("Failed to create output file");
            let mut writer = BufWriter::new(file);
            let report_interval = (count / 10).max(1);

            for i in 0..count {
                let mut nonce = [0u8; 12];
                rng.fill(&mut nonce);
                let block = ChaCha::block_for(key, nonce, 0, rounds);
                // Record: [nonce: 12 bytes][block_output: 64 bytes]
                writer.write_all(&nonce).unwrap();
                writer.write_all(&block).unwrap();

                if (i + 1) % report_interval == 0 {
                    eprintln!("Progress: {}/{} ({:.0}%)", i + 1, count,
                              (i + 1) as f64 / count as f64 * 100.0);
                }
            }
            writer.flush().unwrap();
            let mb = count as f64 * 76.0 / (1024.0 * 1024.0);
            eprintln!("Done! {} ({:.1} MB)", filename, mb);
            eprintln!("Key to reproduce: {}", bytes_to_hex(&key));
        }

        _ => {
            eprintln!("Unknown mode '{}'. Use: stream | counter | nonce", mode);
            std::process::exit(1);
        }
    }
}
