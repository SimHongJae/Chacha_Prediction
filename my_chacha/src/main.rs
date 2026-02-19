use rand::Rng;
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

        for _ in 0..self.rounds / 2 {
            Self::quarter_round(&mut working_state, 0, 4, 8, 12);
            Self::quarter_round(&mut working_state, 1, 5, 9, 13);
            Self::quarter_round(&mut working_state, 2, 6, 10, 14);
            Self::quarter_round(&mut working_state, 3, 7, 11, 15);

            Self::quarter_round(&mut working_state, 0, 5, 10, 15);
            Self::quarter_round(&mut working_state, 1, 6, 11, 12);
            Self::quarter_round(&mut working_state, 2, 7, 8, 13);
            Self::quarter_round(&mut working_state, 3, 4, 9, 14);
        }

        let mut output = [0u32; 16];
        for i in 0..16 {
            output[i] = working_state[i].wrapping_add(self.state[i]);
        }

        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }

        output
    }
}

/// Binary format:
///   For each sequence:
///     [num_words: u32] [word0: u32] [word1: u32] ... [wordN: u32]
///   num_words=0 marks end of file (not written, just EOF)
///
/// This way Python knows where each sequence starts/ends.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <rounds> <num_sequences> <blocks_per_seq>", args[0]);
        eprintln!("Example: {} 4 100000 4", args[0]);
        eprintln!("  -> 100K sequences, each with 4 blocks (64 u32 words)");
        std::process::exit(1);
    }

    let rounds: u32 = args[1].parse().expect("Invalid rounds");
    let num_sequences: usize = args[2].parse().expect("Invalid num_sequences");
    let blocks_per_seq: usize = args[3].parse().expect("Invalid blocks_per_seq");

    assert!(rounds % 2 == 0, "Rounds must be even");

    let words_per_seq = blocks_per_seq * 16; // 16 u32 per block
    let mut rng = rand::thread_rng();

    let dataset_dir = "../dataset";
    fs::create_dir_all(dataset_dir).expect("Failed to create dataset directory");

    let filename = format!("{}/chacha{}_seq.bin", dataset_dir, rounds);
    let file = File::create(&filename).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut total_words: usize = 0;

    for seq_idx in 0..num_sequences {
        // Random key and nonce for each sequence
        let mut key = [0u8; 32];
        let mut nonce = [0u8; 8];
        rng.fill(&mut key);
        rng.fill(&mut nonce);

        let mut chacha = ChaCha::new(key, nonce, rounds);

        // Write sequence length header
        writer.write_all(&(words_per_seq as u32).to_le_bytes()).unwrap();

        // Write all words for this sequence
        for _ in 0..blocks_per_seq {
            let output = chacha.next_block();
            for &word in &output {
                writer.write_all(&word.to_le_bytes()).unwrap();
            }
        }

        total_words += words_per_seq;

        if (seq_idx + 1) % 10_000 == 0 {
            eprintln!(
                "Progress: {}/{} sequences ({} words)",
                seq_idx + 1,
                num_sequences,
                total_words
            );
        }
    }

    writer.flush().unwrap();

    let file_size_mb = ((num_sequences * (1 + words_per_seq)) * 4) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "Done! {} created ({} sequences x {} words = {} total, {:.1} MB)",
        filename, num_sequences, words_per_seq, total_words, file_size_mb
    );
}
