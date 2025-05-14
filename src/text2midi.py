import os
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
from transformers import AutoTokenizer

ignored_tokens = {"[BOS]", "[UNK]", "[PAD]", "[EOS]", "[MASK]", "[NONE]", "Track_Skyline", "Track_Midi"}  # Define tokens to skip

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["[PAD]", "[BOS]", "[EOS]", "[MASK]"],  # Adjusted for Hugging Face compatibility
    "use_chords": True,
    "chord_maps": {
        "+" : (0, 4, 8), "/o7" : (0, 3, 6, 10), "7" : (0, 4, 7, 10),
        "M" : (0, 4, 7), "M7" : (0, 4, 7, 11), "m" : (0, 3, 7),
        "m7" : (0, 3, 7, 10), "o" : (0, 3, 6), "o7" : (0, 3, 6, 9),
        "sus2" : (0, 2, 7), "sus4" : (0, 5, 7)
    },
    "chord_tokens_with_root_note": True,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # Number of tempo bins
    "tempo_range": (50, 180),  # (min, max)
}

config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

