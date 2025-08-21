"""
Filter for .csv output of unscrapr.py
"""
import pandas as pd
import re
from typing import Iterable, Union, Optional

# Define Scrabble letter scores
scrabble_scores = {
    'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1,
    'j': 8, 'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1,
    's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
}

def calculate_scrabble_score(phrase):
    """Calculates the Scrabble score for a string, ignoring non-letters."""
    score = 0
    for char in str(phrase).lower():
        score += scrabble_scores.get(char, 0)  
    return score


def _load_wordlists(wordlist_paths: Union[str, Iterable[str]]) -> set:
    """
    Load one or many wordlist files 
    Each line is parsed up to the first ';', lowercased, non-letters stripped.
    Returns a set of processed entries.
    """
    if isinstance(wordlist_paths, (str, bytes)):
        paths = [wordlist_paths]
    else:
        paths = list(wordlist_paths)

    combined = set()
    for path in paths:
        print(f"Reading wordlist from '{path}'...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if ';' not in line:
                    continue
                token = re.sub(r'[^a-z]', '', line.strip().split(';')[0].lower())
                if token:
                    combined.add(token)
    print(f"Loaded {len(combined)} processed words from {len(paths)} file(s).")
    return combined


def _load_hits(hits_paths: Union[str, Iterable[str]]) -> set:
    """
    Load one or many 'hits' files. Each non-empty, non-comment line becomes an EXACT
    string to exclude (trimmed). No normalization beyond strip().
    """
    if isinstance(hits_paths, (str, bytes)):
        paths = [hits_paths]
    else:
        paths = list(hits_paths)

    hits = set()
    for path in paths:
        print(f"Reading hits from '{path}'...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                hits.add(s)
    print(f"Loaded {len(hits)} exact hits from {len(paths)} file(s).")
    return hits


def filter_and_refine_csv(csv_file_path,
                          wordlist_file_paths: Union[str, Iterable[str]],
                          output_file_path,
                          csv_column_name='phrase',
                          hits_file_paths: Optional[Union[str, Iterable[str]]] = None):
    """
    Reads a CSV, filters it against one or many wordlists, applies additional rules

    Parameters
    ----------
    csv_file_path : str
        Input CSV path (must contain a column named `csv_column_name`).
    wordlist_file_paths : str | Iterable[str]
     output_file_path : str
        Where to write final phrases (one per line).
    csv_column_name : str
        Column name in CSV containing phrases (default: 'phrase').
    hits_file_paths : str | Iterable[str] | None
        One or many files with EXACT phrases (trimmed lines) to exclude.
    """
    try:
        # Initial Filtering Against Wordlists
        wordlist = _load_wordlists(wordlist_file_paths)

        print(f"Reading CSV file from '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)
        print("Successfully loaded CSV file.")

        print("Performing initial comparison against wordlist(s)...")
        unmatched_phrases = []
        for phrase in df[csv_column_name]:
            processed_phrase = re.sub(r'[^a-z]', '', str(phrase).lower())
            if processed_phrase not in wordlist:
                unmatched_phrases.append(phrase)
        print(f"Found {len(unmatched_phrases)} phrases not in the wordlist(s). Now applying advanced filtering...")

        # Apply  Filtering Rules 
        allowed_short_words = {'a', 'i', 'on', 'or', 'at', 'to', 'in', 'it', 'so', 'be', 'of'}
        excluded_start_words = {'and', 'be', 'to', 'is', 'the', 'he', 'she', 'st', 'a', 'of', 'which', 'excerpt', 'than', 'those', 'happened', 'was', 'are', 'winny'
                                'growing', 'likely', 'even', 'scheme', 'but', 'killed',
                                'have', 'will', 'know', 'saying'}
        excluded_end_words = {'to', 'is', 'the', 'he', 'she', 'st', 'a', 'her', 'be', 'him', 'for',
                              'of', 'than', 'how', 'when', 'don', 'i', 'or', 'so', 'and',
                              'who', 'previously', 'wouldn', 'couldn', 'shouldn', }
        excluded_words = {'cnn', 'digg', 'del', 'icio', 'reddit', 'llc', 'lake', 'bbc', 'zur',
                          'verf', 'gung', 'der', 'das', 'iraq', 'bush', 'rice', 'trump', 'guan',
                          'cnnmoney', 'com', 'health', 'crime', 'tech', 'community', 'firefox',
                          'muhammed', 'jew', 'jewish', 'hezbollah', 'menzies', 'chertoff',
                          'john', 'michael', 'iraqi', 'effectively', 'approximately',
                          'exclusive', 'phosphorus', 'alex', 'sex', 'jones', 'msdn',
                          'archive', 'york', 'microsoft', 'yahoo', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'janurary', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'xml', 'republicans', 'obama', 'osama', 'mccain', 'gore'}

        final_filtered_phrases = []
        for phrase in unmatched_phrases:
            words = str(phrase).lower().split()
            if not words:
                continue

            is_valid = True

            # Excluded start/end words
            if words[0] in excluded_start_words:
                is_valid = False
            if is_valid and words[-1] in excluded_end_words:
                is_valid = False

            # Excluded anywhere
            if is_valid:
                for word in words:
                    if word in excluded_words:
                        is_valid = False
                        break

            # RInvalid short words 
            
            if is_valid:
                for word in words:
                    cleaned_word = re.sub(r'[^a-z]', '', word)
                    if len(cleaned_word) <= 2 and cleaned_word not in allowed_short_words:
                        is_valid = False
                        break

            if is_valid:
                final_filtered_phrases.append(phrase)

        # Exclude prev hits or blacklist
        if hits_file_paths:
            hits = _load_hits(hits_file_paths)
            before = len(final_filtered_phrases)
            final_filtered_phrases = [p for p in final_filtered_phrases if str(p).strip() not in hits]
            removed = before - len(final_filtered_phrases)
            print(f"Excluded {removed} phrase(s) present in hits file(s).")

        # --- Stage 3: Write Final Output ---
        final_filtered_phrases.sort(key=calculate_scrabble_score, reverse=True)
        print(f"Advanced filtering complete. {len(final_filtered_phrases)} phrases remain.")
        print(f"Writing final list to '{output_file_path}'...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for phrase in final_filtered_phrases:
                f.write(f"{phrase}\n")

        print(f"\nProcessing complete! Final output saved to '{output_file_path}'.")

    except FileNotFoundError as e:
        print(f"Error: The file {e.filename} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- How to Use ---
csv_file = 'your_csv_here.csv'

# Wordlists you want to exclude matches from
wordlist_files = ['xwi.txt', 'personal_wordlist.txt', 'spreadthewordlist.txt' ]

# Optional: one or many exact-match hits files to exclude from output (same format as output)
hits_files = ['hits.txt', 'exclude.txt']  

output_file = 'output_file.txt'

filter_and_refine_csv(csv_file_path=csv_file,
                      wordlist_file_paths=wordlist_files,
                      output_file_path=output_file,
                      csv_column_name='phrase',
                      hits_file_paths=hits_files)
