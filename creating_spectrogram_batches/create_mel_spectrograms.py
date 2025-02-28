#!/usr/bin/env python
import os
import gc
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import h5py

# ---------------- Data Loading Functions ----------------
def load_df(file_path):
    """Load a CSV or Excel dataset."""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)

def load_downloaded_mapping(mapping_path):
    """Load the downloaded mapping from a JSON file."""
    with open(mapping_path, 'r') as fp:
        return json.load(fp)

# ---------------- Data Preparation ----------------
# Paths to your dataset files.
excel_file_path = "../../datasets/Greek_Music_Dataset.xlsx"
csv_file_path = "../../datasets/Dataset.csv"
mapping_path = '../../downloaded_mapping_wav_updated.json'

# Load datasets.
df_excel = load_df(excel_file_path)
df_csv = load_df(csv_file_path)

# Choose one dataset. (Adjust as needed.)
df = df_excel.copy()

# Load the downloaded mapping.
downloaded_mapping = load_downloaded_mapping(mapping_path)

# Mark downloaded songs and create a local filename column.
df['downloaded'] = df["YouTube Link"].apply(lambda x: x in downloaded_mapping)
df = df[df['downloaded']].copy()
print(f"Found {len(df)} downloaded songs in the dataset.")
df['local_filename'] = df["YouTube Link"].map(downloaded_mapping)

# ---------------- Audio Processing Helpers ----------------
def trim_silence(y, sr, top_db=30):
    """Trim silence from an audio signal."""
    return librosa.effects.trim(y, top_db=top_db)[0]

def segment_audio(y, sr, window_duration=30, hop_duration=15):
    """
    Segment audio into overlapping segments.
    
    Returns a list of audio segments.
    """
    window_length = int(window_duration * sr)
    hop_length_samples = int(hop_duration * sr)
    segments = []
    for start in range(0, len(y) - window_length + 1, hop_length_samples):
        segments.append(y[start:start + window_length])
    return segments

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Compute the mel spectrogram in decibel scale."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel_spec, ref=np.max).tolist()

# ---------------- HDF5 Write Helper ----------------
def append_songs_to_h5(h5_file_path, new_entries):
    """
    Append new song entries to an HDF5 file.
    new_entries is a dict mapping song names to song data (dict with keys "song", "segments", "target").
    Each song is stored as a group; within each group, a dataset "target" is created (as a 1D array)
    and a subgroup "segments" is created to store each segment as a dataset.
    """
    output_dir = os.path.dirname(h5_file_path)
    if output_dir:  # Only create directory if output_dir is non-empty
        os.makedirs(output_dir, exist_ok=True)
    try:
        with h5py.File(h5_file_path, 'a') as f:
            for song_name, song_data in new_entries.items():
                if song_name in f:
                    print(f"Song {song_name} already exists in {h5_file_path}; skipping.")
                    continue
                grp = f.create_group(song_name)
                # Save target vector as a dataset.
                grp.create_dataset("target", data=np.array(song_data["target"], dtype='int8'))
                # Create a subgroup for segments.
                seg_grp = grp.create_group("segments")
                for i, seg in enumerate(song_data["segments"]):
                    # Convert seg to np.array of float32.
                    seg_arr = np.array(seg, dtype='float32')
                    seg_grp.create_dataset(f"segment_{i}", data=seg_arr, compression="gzip")
                print(f"Appended song '{song_name}' with {len(song_data['segments'])} segments.")
    except OSError as e:
        print(f"Failed to write to {h5_file_path}: {e}")
        return

# ---------------- Batched Processing Function ----------------
def process_dataset_batched(
    df,
    audio_folders,              # List of folders where the audio files reside.
    sr=22050,
    window_duration=30,         # Seconds per segment.
    hop_duration=15,            # Seconds between segments.
    num_splits=16,              # Number of batches.
    song_key_column="Song",
    genre_order=None,           # List of genre column names in a fixed order.
    output_file_base="mel_spectrogram_batches/data.h5",
    file_limit=None             # Process only this many songs from df.
):
    """
    Processes a DataFrame subset in batches and appends new song entries to an HDF5 file.
    
    For each song:
      - Locates the audio file (appending '.wav' if needed).
      - Loads the full audio, trims silence, segments it, and computes mel spectrograms.
      - Builds a target vector from genre flags.
    
    New entries (a dict mapping song name to its song dictionary) are appended to the HDF5 file.
    
    Returns:
      dataset: List of song dictionaries.
      song_map: Dict mapping song names to their list of mel spectrogram segments.
      processed_song_names: List of processed song names.
    """
    if file_limit is not None:
        df = df.head(file_limit)
    if "downloaded" in df.columns:
        df = df[df["downloaded"] == True]
    
    df_batches = np.array_split(df, num_splits)
    dataset = []
    song_map = {}
    processed_song_names = []
    new_entries = {}
    
    for i, batch in enumerate(df_batches):
        print(f"Processing batch {i+1}/{num_splits} - {len(batch)} songs...")
        for idx, row in batch.iterrows():
            song_name = row[song_key_column]
            candidate_file = song_name if song_name.lower().endswith(".wav") else song_name + ".wav"
            file_found = False
            file_path = None
            for folder in audio_folders:
                candidate_path = os.path.join(folder, candidate_file)
                if os.path.exists(candidate_path):
                    file_found = True
                    file_path = candidate_path
                    break
            if not file_found:
                print(f"File not found for '{song_name}' (tried '{candidate_file}')")
                continue
            
            target_vector = []
            if genre_order:
                for genre in genre_order:
                    val = row[genre]
                    is_true = (val.strip().lower() in ("yes", "true", "1")) if isinstance(val, str) else bool(val)
                    target_vector.append(1 if is_true else 0)
            
            try:
                y, sr = librosa.load(file_path, sr=sr)
                y = trim_silence(y, sr, top_db=30)
                segments = segment_audio(y, sr, window_duration, hop_duration)
                seg_specs = [compute_mel_spectrogram(seg, sr) for seg in segments]
            except Exception as e:
                print(f"Error processing '{file_path}': {e}")
                continue
            
            song_dict = {
                "song": song_name,
                "segments": seg_specs,
                "target": target_vector
            }
            dataset.append(song_dict)
            song_map[song_name] = seg_specs
            processed_song_names.append(song_name)
            new_entries[song_name] = song_dict
            print(f"Processed '{song_name}' with {len(seg_specs)} segments.")
            del y, segments, seg_specs
            gc.collect()
        del batch
        gc.collect()
    
    # Append new entries to the HDF5 file.
    append_songs_to_h5(output_file_base, new_entries)
    print(f"Appended {len(new_entries)} new entries to '{output_file_base}'.")
    
    return dataset, song_map, processed_song_names

# ---------------- Combined Processing for Train/Test ----------------
def process_and_split_dataset(
    df,
    audio_folders,
    
    # Audio processing parameters
    sr=22050,
    window_duration=30,
    hop_duration=15,
    
    # Train/test split parameters
    train_ratio=0.8,
    train_batches=6,
    test_batches=2,
    
    # Data and output configuration
    song_key_column="Song",
    genre_order=None,
    # paths for the temporary train and test data files holding the data processed in the current run
    train_h5_path="train_data.h5",
    test_h5_path="test_data.h5",
    file_limit=None, # song files to process per script execution
    processed_file="processed_songs.json" # the already processed songs
):
    """
    Splits the DataFrame into train and test sets (keeping original order),
    filters out already processed songs (loaded from processed_file),
    and then processes each subset in batches, writing the results to HDF5 files.
    
    After processing, the processed song names are updated in processed_file.
    
    Returns:
      train_data, test_data: Lists of song dictionaries.
      train_song_map, test_song_map: Mappings of song names to spectrogram segments.
      train_song_names, test_song_names: Lists of processed song names.
    """
    import json
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as pf:
            try:
                processed_songs = json.load(pf)
            except json.decoder.JSONDecodeError:
                processed_songs = []
    else:
        processed_songs = []
    
    df = df[~df[song_key_column].isin(processed_songs)]
    
    if file_limit is not None:
        df = df.head(file_limit)
    
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Total songs to process this run: {len(df)}. Training: {len(train_df)}. Testing: {len(test_df)}.")
    
    print("Processing training data...")
    train_data, train_song_map, train_song_names = process_dataset_batched(
        df=train_df,
        audio_folders=audio_folders,
        sr=sr,
        window_duration=window_duration,
        hop_duration=hop_duration,
        num_splits=train_batches,
        song_key_column=song_key_column,
        genre_order=genre_order,
        output_file_base=train_h5_path
    )
    
    print("Processing testing data...")
    test_data, test_song_map, test_song_names = process_dataset_batched(
        df=test_df,
        audio_folders=audio_folders,
        sr=sr,
        window_duration=window_duration,
        hop_duration=hop_duration,
        num_splits=test_batches,
        song_key_column=song_key_column,
        genre_order=genre_order,
        output_file_base=test_h5_path
    )
    
    # Update processed songs list.
    all_processed = processed_songs + train_song_names + test_song_names
    with open(processed_file, 'w') as pf:
        json.dump(all_processed, pf, indent=2)
    print(f"Updated processed songs list saved to '{processed_file}'.")
    
    return train_data, test_data, train_song_map, test_song_map, train_song_names, test_song_names

# ---------------- Main Script Execution ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process mel spectrogram batches for Greek Music Dataset (HDF5 version)")
    parser.add_argument("--file_limit", type=int, default=None, help="Number of new songs to process in this run")
    args = parser.parse_args()
    
    audio_folders = ["../../../downloads_wav_1", "../../../downloads_wav_2"]
    genre_order = ["LAIKO", "REMPETIKO", "ENTEXNO", "ROCK", "Mod LAIKO", "POP", "ENALLAKTIKO", "HIPHOP/RNB"]
    
    train_data, test_data, train_song_map, test_song_map, train_song_names, test_song_names = process_and_split_dataset(
        df=df,
        audio_folders=audio_folders,
        sr=22050,
        window_duration=30,
        hop_duration=15,
        train_ratio=0.8,
        train_batches=16,
        test_batches=5,
        song_key_column="Song",
        genre_order=genre_order,
        train_h5_path="train_data.h5",
        test_h5_path="test_data.h5",
        file_limit=args.file_limit,
        processed_file="processed_songs.json"
    )
    
    # Optionally, you can also update additional metadata in JSON files if needed.
    # For example, store the list of processed song names:
    with open('train_song_names.json', 'w') as f:
        json.dump(train_song_names, f, indent=2)
    print("train_song_names saved to 'train_song_names.json'.")
    
    with open('test_song_names.json', 'w') as f:
        json.dump(test_song_names, f, indent=2)
    print("test_song_names saved to 'test_song_names.json'.")
    
    with open('genre_order.json', 'w') as f:
        json.dump(genre_order, f, indent=2)
    print("genre_order saved to 'genre_order.json'.")