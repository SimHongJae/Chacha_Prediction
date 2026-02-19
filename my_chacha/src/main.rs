use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

struct ChaCha {
    state: [u32; 16],
    rounds: u32,
}

impl ChaCha {
    pub fn new(key: [u8; 32], nonce: [u8; 8], rounds: u32) -> Self {
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

        // Counter
        s[12] = 0;
        s[13] = 0;

        // Nonce (8 bytes -> 2 u32s)
        s[14] = u32::from_le_bytes(nonce[0..4].try_into().unwrap());
        s[15] = u32::from_le_bytes(nonce[4..8].try_into().unwrap());

        ChaCha { state: s, rounds }
    }

    fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(16);
        s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(12);
        s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(8);
        s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(7);
    }

    pub fn next_block(&mut self) -> [u32; 16] {
        let mut working_state = self.state;

        // Double rounds = rounds / 2
        for _ in 0..self.rounds / 2 {
            // Column rounds
            Self::quarter_round(&mut working_state, 0, 4, 8, 12);
            Self::quarter_round(&mut working_state, 1, 5, 9, 13);
            Self::quarter_round(&mut working_state, 2, 6, 10, 14);
            Self::quarter_round(&mut working_state, 3, 7, 11, 15);

            // Diagonal rounds
            Self::quarter_round(&mut working_state, 0, 5, 10, 15);
            Self::quarter_round(&mut working_state, 1, 6, 11, 12);
            Self::quarter_round(&mut working_state, 2, 7, 8, 13);
            Self::quarter_round(&mut working_state, 3, 4, 9, 14);
        }

        // Feed-forward: add initial state
        let mut output = [0u32; 16];
        for i in 0..16 {
            output[i] = working_state[i].wrapping_add(self.state[i]);
        }

        // Increment block counter
        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }

        output
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <rounds> <num_u32_words>", args[0]);
        eprintln!("Example: {} 4 4000000", args[0]);
        std::process::exit(1);
    }

    let rounds: u32 = args[1].parse().expect("Invalid rounds number");
    let num_words: usize = args[2].parse().expect("Invalid word count");

    assert!(rounds % 2 == 0, "Rounds must be even (2, 4, 6, 8, ...)");

    // Fixed key and nonce for reproducibility
    let key: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let nonce: [u8; 8] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

    let mut chacha = ChaCha::new(key, nonce, rounds);

    // Create dataset directory
    let dataset_dir = "../dataset";
    fs::create_dir_all(dataset_dir).expect("Failed to create dataset directory");

    let filename = format!("{}/chacha{}_seq.bin", dataset_dir, rounds);
    let file = File::create(&filename).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let blocks_needed = (num_words + 15) / 16; // 16 u32 words per block
    let mut words_written: usize = 0;

    for block_idx in 0..blocks_needed {
        let output = chacha.next_block();

        for &word in &output {
            if words_written >= num_words {
                break;
            }
            writer.write_all(&word.to_le_bytes()).unwrap();
            words_written += 1;
        }

        if (block_idx + 1) % 100_000 == 0 {
            eprintln!(
                "Progress: {} blocks, {} words written",
                block_idx + 1,
                words_written
            );
        }
    }

    writer.flush().unwrap();

    let file_size_mb = (words_written * 4) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "Done! {} created ({} u32 words, {:.1} MB)",
        filename, words_written, file_size_mb
    );
}
