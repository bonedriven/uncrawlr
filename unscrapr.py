"""
Unscrapr – n-gram miner for AI corpora.

This script streams text from large corpora, identifies candidate n-grams (2- to 5-word phrases),
and scores them based on a combination of statistical measures and qualitative heuristics
to discover novel phrases.

-- Dependencies --
- Required: datasets, ftfy, tqdm, langid
- For local OWT2: zstandard
- For advanced scoring: spacy, jellyfish
 - A medium spaCy model (en_core_web_md) is needed for semantic scoring.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import pathlib
import re
import sys
import tarfile
import gc
import itertools
import contextlib
try:
   import duckdb
except Exception:
   duckdb = None
from collections import Counter, defaultdict
from typing import TextIO, Dict, List, Tuple, Any
import multiprocessing as mp
import os
import pickle

import ftfy
import langid
from datasets import load_dataset
from tqdm import tqdm

# Attempt to import optional dependencies
try:
   import zstandard as zstd
except ImportError:
   zstd = None

try:
   import spacy
   from spacy.tokens import Doc
except ImportError:
   spacy = None
   Doc = None

try:
   import jellyfish
except ImportError:
   jellyfish = None

# -----------------------------
# Constants & Regexes
# -----------------------------
WORD_RE = re.compile(r"[A-Za-z]+")
MULTISPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.I)
CODE_SNIPPET_RE = re.compile(r"[;{}<>]{2,}|class\s+\w+|def\s+\w+|\bimport\b|\bvar\s+\w+", re.I)
SEO_SPAM_RE = re.compile(r"free\s+shipping|best\s+price|coupon|promo\s+code|buy\s+now", re.I)
REPEAT_SYLLABLE_RE = re.compile(r"\b([a-z]{2,4})\1{2,}\b")
PUNC_RATIO_MAX = 0.15
UPPER_RUN_MAX = 0.70

# Stopword sets for filtering
HARD_STOP = set()
SOFT_STOP = set()
LIGHT_STOP = set()
ALL_STOPS = HARD_STOP | SOFT_STOP | LIGHT_STOP
EDGE_STOPS = SOFT_STOP | LIGHT_STOP | set()


# -----------------------------
# Text Processing & Filtering Helpers
# -----------------------------

def normalize(text: str) -> str:
   """Cleans and standardizes raw text."""
   text = ftfy.fix_text(text)
   text = text.replace("\u00A0", " ")
   text = MULTISPACE_RE.sub(" ", text).strip()
   return text

def is_reasonable(text: str) -> bool:
   """Performs a series of quick checks to discard low-quality documents."""
   if len(text) < 80: return False
   if URL_RE.search(text) and text.count("http") > 1: return False
   if CODE_SNIPPET_RE.search(text): return False
   if SEO_SPAM_RE.search(text): return False
   
   letters = sum(c.isalpha() for c in text)
   if letters == 0: return False
   
   punc = sum(not c.isalnum() and not c.isspace() for c in text)
   if punc / max(1, letters + punc) > PUNC_RATIO_MAX: return False
   
   uppers = sum(c.isupper() for c in text if c.isalpha())
   if uppers / max(1, letters) > UPPER_RUN_MAX: return False
   
   if REPEAT_SYLLABLE_RE.search(text.lower()): return False
   
   return True

def tokenize(text: str, min_len: int, max_len: int) -> list[str]:
   """Extracts and lowercases alphabetic tokens within a specified length range."""
   raw_tokens = WORD_RE.findall(text)
   return [t.lower() for t in raw_tokens if min_len <= len(t) <= max_len]

def ngrams(seq: list[str], n: int):
   """Yields n-gram tuples from a sequence."""
   for i in range(len(seq) - n + 1):
       yield tuple(seq[i:i + n])

def accept_tokens(toks: tuple[str, ...], args: argparse.Namespace) -> bool:
   """Applies filtering rules to an n-gram tuple."""
   content_words = [t for t in toks if t not in ALL_STOPS and len(t) >= 3]
   if not content_words:
       return False

   if args.no_edge_stops and (toks[0] in EDGE_STOPS or toks[-1] in EDGE_STOPS):
       return False

   if args.min_content_ratio > 0 and (len(content_words) / len(toks)) < args.min_content_ratio:
       return False

   if not args.allow_stopword_runs:
       stopword_run = 0
       for t in toks:
           stopword_run = stopword_run + 1 if t in ALL_STOPS else 0
           if stopword_run >= 3:
               return False
               
   return True

# -----------------------------
# Statistical & Association Measures
# -----------------------------

def zipf(count: int, total: int) -> float:
   """Calculates the Zipf frequency of a count."""
   if count <= 0 or total <= 0: return 0.0
   return round(math.log10((count / total) * 1_000_000_000 + 1e-12), 3)

def npmi_bigram(a: str, b: str, uni: Counter, bi: Counter, T_tokens: int, T_bi: int) -> float:
   """Calculates Normalized Pointwise Mutual Information for a bigram."""
   pa = uni.get(a, 0) / T_tokens if T_tokens else 0.0
   pb = uni.get(b, 0) / T_tokens if T_tokens else 0.0
   pab = bi.get((a, b), 0) / T_bi if T_bi else 0.0
   if min(pa, pb, pab) <= 0: return -1.0
   pmi = math.log(pab / (pa * pb))
   return pmi / (-math.log(pab))

def llr_bigram(a: str, b: str, uni: Counter, bi: Counter, T_tokens: int) -> float:
   """Calculates the Log-Likelihood Ratio for a bigram."""
   k11 = bi.get((a, b), 0)
   k12 = max(1, uni.get(a, 0) - k11)
   k21 = max(1, uni.get(b, 0) - k11)
   k22 = max(1, T_tokens - (k11 + k12 + k21))

   def entropy(k_list):
       n = sum(k_list)
       return sum(k * math.log(k / n if k > 0 else 1) for k in k_list)

   llr = 2 * (entropy([k11, k12, k21, k22]) - 
              entropy([k11 + k12, k21 + k22]) - 
              entropy([k11 + k21, k12 + k22]))
   return max(0.0, llr)


def assoc_bridge(kind_len: int, toks: tuple[str, ...], score_type: str, uni: Counter, bi: Counter, T_tokens: int, T_bi: int) -> float:
   """Calculates the minimum association score between adjacent words in a phrase."""
   if kind_len < 2: return -1.0
   
   if score_type == 'llr':
       vals = [llr_bigram(toks[i], toks[i + 1], uni, bi, T_tokens) for i in range(kind_len - 1)]
   else: # default to npmi
       vals = [npmi_bigram(toks[i], toks[i + 1], uni, bi, T_tokens, T_bi) for i in range(kind_len - 1)]
       
   return min(vals) if vals else -1.0

# -----------------------------
# Novelty & Qualitative Scoring
# -----------------------------

def score_semantic_surprise(doc: Doc) -> float:
   """Scores a phrase based on the semantic distance between its words."""
   if not doc or not doc.has_vector or not len(doc.vocab.vectors): return 0.0
   
   tokens = [t for t in doc if t.has_vector and not t.is_oov and not t.is_stop]
   if len(tokens) < 2: return 0.0
   
   similarities = [tokens[i].similarity(tokens[i+1]) for i in range(len(tokens) - 1)]
   if not similarities: return 0.0
       
   return 1.0 - (sum(similarities) / len(similarities))

def score_phonetic_appeal(phrase_tuple: tuple[str, ...]) -> float:
   """Scores a phrase based on alliteration."""
   if not jellyfish or len(phrase_tuple) < 2: return 0.0
       
   codes = [jellyfish.metaphone(t) for t in phrase_tuple]
   if not codes[0]: return 0.0
       
   alliteration_count = sum(1 for code in codes[1:] if code == codes[0])
   return alliteration_count / (len(phrase_tuple) - 1)

def scrabble_word_score(word: str) -> int:
   table = {**{c:1 for c in "AEILNORSTU"}, **{c:2 for c in "DG"}, **{c:3 for c in "BCMP"},
             **{c:4 for c in "FHVWY"}, "K":5, **{c:8 for c in "JX"}, **{c:10 for c in "QZ"}}
   s = 0
   for ch in word.upper():
       if ch == "'":  # ignore apostrophes
           continue
       s += table.get(ch, 0)
   return s

def score_scrabble(phrase_tuple: tuple[str, ...]) -> float:
   if not phrase_tuple: return 0.0
   return sum(scrabble_word_score(w) for w in phrase_tuple) / max(1, len(phrase_tuple))


def is_grammatically_interesting(doc: Doc) -> bool:
   """Applies stricter POS-based rules to filter out boring grammatical structures."""
   if not doc: return True
   tokens = list(doc)
   pos_tags = [t.pos_ for t in tokens]
   lemmas = [t.lemma_ for t in tokens]

   if not any(p in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ"} for p in pos_tags): return False
   if pos_tags[0] in {"SCONJ", "ADP"} and len(tokens) > 2 and pos_tags[1] in {"PRON", "DET"}: return False

   LOW_CONTENT_VERBS = {"be", "have", "do", "know", "think", "want", "need", "say", "go", "get", "make", "see"}
   if len(tokens) > 2 and pos_tags[0] == "PRON" and (lemmas[1] in LOW_CONTENT_VERBS or pos_tags[1] == "AUX"): return False

   if all(p in {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ", "PUNCT"} for p in pos_tags): return False
       
   return True

# -----------------------------
# Corpus Streaming
# -----------------------------

def stream_owt2_path(tar_path: str, max_shards: int | None, max_bytes: int | None):
   """Streams and decompresses a local OpenWebText2 TAR file."""
   if not zstd: raise RuntimeError("The 'zstandard' library is required for --owt2-path.")
   
   with tarfile.open(tar_path, mode="r") as outer_tar:
       members = outer_tar.getmembers()
       processed = 0
       for member in members:
           if not member.name.endswith(".jsonl.zst"): continue
           
           if max_shards is not None and processed >= max_shards:
               break
           processed += 1
           
           fobj = outer_tar.extractfile(member)
           if not fobj: continue
           
           dctx = zstd.ZstdDecompressor()
           with dctx.stream_reader(fobj) as reader:
               buf = io.BufferedReader(reader)
               bytes_read = 0
               while True:
                   line = buf.readline()
                   if not line: break
                   if max_bytes and bytes_read >= max_bytes: break
                   bytes_read += len(line)
                   try:
                       yield json.loads(line).get("text", "")
                   except (json.JSONDecodeError, AttributeError):
                       continue

def stream_owt2_uncompressed_dir(dir_path: str):
   """Streams from a directory of uncompressed .jsonl files."""
   jsonl_files = list(pathlib.Path(dir_path).glob('**/*.jsonl'))
   if not jsonl_files:
       raise FileNotFoundError(f"No .jsonl files found in directory: {dir_path}")
   
   print(f"[info] Found {len(jsonl_files)} .jsonl files to process.", file=sys.stderr)
   for file_path in jsonl_files:
       with open(file_path, 'r', encoding='utf-8') as f:
           for line in f:
               try:
                   yield json.loads(line).get("text", "")
               except (json.JSONDecodeError, AttributeError):
                   continue

def stream_corpus(args: argparse.Namespace):
   """Yields text from the configured corpus."""
   if args.corpus == "owt2-local":
       if not args.owt2_path: raise ValueError("--owt2-path is required for --corpus owt2-local")
       yield from stream_owt2_path(args.owt2_path, args.owt2_max_shards, args.owt2_max_bytes)
       return
   # -- REVISION: Add new corpus type for uncompressed directory
   if args.corpus == "owt2-dir":
       if not args.owt2_dir_path: raise ValueError("--owt2-dir-path is required for --corpus owt2-dir")
       yield from stream_owt2_uncompressed_dir(args.owt2_dir_path)
       return

   ds_map = {
       "c4": ("c4", args.hf_config or "en"), "oscar": ("oscar-corpus/OSCAR-2301", args.hf_config or "en"),
       "openwebtext": ("openwebtext", None), "fineweb": ("HuggingFaceFW/fineweb", args.hf_config),
       "fineweb-edu": ("HuggingFaceFW/fineweb-edu", args.hf_config),
   }
   if args.corpus not in ds_map: raise ValueError(f"Unsupported corpus '{args.corpus}'")
       
   ds_path, ds_config = ds_map[args.corpus]
   ds = load_dataset(ds_path, ds_config, split=args.split, streaming=True)
   for row in ds:
       yield row.get("text", "")

# -----------------------------
# Main Script Logic Functions
# -----------------------------

def setup_args() -> argparse.Namespace:
   """Sets up and parses command-line arguments."""
   ap = argparse.ArgumentParser(description="Stream corpora to find novel, high-quality phrases.")
   
   # Corpus config
   ap.add_argument("--corpus", choices=["owt2-local", "owt2-dir", "fineweb", "fineweb-edu", "openwebtext", "oscar", "c4"], required=True)
   ap.add_argument("--hf-config", help="HF dataset config (e.g., 'en' for c4)")
   ap.add_argument("--split", default="train", help="Dataset split to use")
   
   # Processing limits
   ap.add_argument("--max-docs", type=int, default=200000, help="Max documents to process")
   ap.add_argument("--max-tokens", type=int, default=5000000000, help="Max tokens to process")
   ap.add_argument("--workers", type=int, default=-1, help="Number of cores to use for scoring (-1 for all)")
   ap.add_argument("--merge-backend", choices=["sqlite-fast","sqlite","lmdb","duckdb","shelve"], default="sqlite-fast", help="On-disk store for merging counters")

   ap.add_argument("--worker-chunk-size", type=int, default=2000, help="Max phrases per worker task to bound memory")
   ap.add_argument("--top-k-per-type", type=int, default=0, help="If >0, only keep/save the top K phrases per n-gram type")
   
   # Pre-filtering
   ap.add_argument("--langid-check", action="store_true", help="Filter out non-English documents")
   ap.add_argument("--dedupe", action="store_true", help="Deduplicate documents based on hash")

   # OWT2 local config
   ap.add_argument("--owt2-path", help="Path to local OpenWebText2 TAR file")
   ap.add_argument("--owt2-dir-path", help="Path to directory of uncompressed OpenWebText2 .jsonl files")
   ap.add_argument("--owt2-max-shards", type=int, help="Max inner shards to process from OWT2 TAR")
   ap.add_argument("--owt2-max-bytes", type=int, help="Max bytes per shard for quick scans")

   # Token & phrase rules
   ap.add_argument("--min-len", type=int, default=1, help="Min token length")
   ap.add_argument("--max-len", type=int, default=15, help="Max token length")
   ap.add_argument("--min-phrase-chars", type=int, default=6, help="Min total letters in a phrase")
   ap.add_argument("--max-phrase-chars", type=int, default=15, help="Max total letters in a phrase")
   ap.add_argument("--allow-stopword-runs", action="store_true", help="Allow phrases with 3+ consecutive stopwords")
   ap.add_argument("--keep-bigrams", action="store_true")
   ap.add_argument("--keep-trigrams", action="store_true")
   ap.add_argument("--keep-fourgrams", action="store_true")
   ap.add_argument("--keep-fivegrams", action="store_true")

   # Scoring & filtering
   ap.add_argument("--score", choices=["npmi", "llr"], default="npmi", help="Association metric to use")
   ap.add_argument("--min-zipf-phrase", type=float, default=3.3, help="Min Zipf score for a phrase")
   ap.add_argument("--pmi-min", type=float, default=0.2, help="Min nPMI association score")
   ap.add_argument("--llr-min", type=float, default=10.0, help="Min LLR association score")
   ap.add_argument("--min-docs", type=int, default=3, help="Min distinct documents a phrase must appear in")
   ap.add_argument("--no-edge-stops", action="store_true", help="Reject phrases starting/ending with stopwords")
   ap.add_argument("--min-content-ratio", type=float, default=0.0, help="Min ratio of content words (0.0=off)")
   ap.add_argument("--pos-filter", action="store_true", help="Use spaCy to filter for interesting grammatical patterns")
   
   # Score weighting
   ap.add_argument("--semantic-weight", type=float, default=0.0, help="Weight for semantic surprise score (0-1)")
   ap.add_argument("--phonetic-weight", type=float, default=0.0, help="Weight for phonetic appeal score (0-1)")
   ap.add_argument("--scrabble-weight", type=float, default=0.0, help="Weight for Scrabble-based interestingness (0-1)")

   # Exclusions & output
   ap.add_argument("--exclude", help="Comma-separated regexes to drop tokens")
   ap.add_argument("--novelty-file", action="append", default=[], help="Path to a list of phrases to exclude")
   ap.add_argument("--wordlist-file", help="Path to personal wordlist (word;score) for exclusion & sparkle profiling")
   ap.add_argument("--summary-file", help="Filename for a single, sorted summary of all novel phrases")
   ap.add_argument("--out-prefix", default="out", help="Prefix for output files")
   ap.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to save/load progress")
   ap.add_argument("--checkpoint-interval", type=int, default=100000, help="Save progress every N documents during corpus scan")
   ap.add_argument("--min-avg-scrabble", type=float, default=1.3, help="Minimum average Scrabble score per word to keep an n-gram")
   
   args = ap.parse_args()
    
   return args

def load_resources(args: argparse.Namespace) -> Dict[str, Any]:
   """Loads all necessary external resources like wordlists and models."""
   total_weight = args.semantic_weight + args.phonetic_weight + args.scrabble_weight
   if total_weight > 1.0: sys.exit("ERROR: Sum of score weights cannot exceed 1.0")
   if args.phonetic_weight > 0 and not jellyfish: sys.exit("ERROR: --phonetic-weight requires 'jellyfish'.")
   if (args.pos_filter or args.semantic_weight > 0) and not spacy: sys.exit("ERROR: This config requires 'spacy'.")

   novelty = set()
   for path in args.novelty_file:
       try:
           with open(path, "r", encoding="utf-8") as f:
               novelty.update(line.strip().lower() for line in f if line.strip())
       except FileNotFoundError:
           print(f"[warn] Novelty file not found: {path}", file=sys.stderr)

   personal_wordlist = set()
   if args.wordlist_file:
       try:
           with open(args.wordlist_file, "r", encoding="utf-8") as f:
               personal_wordlist.update(line.strip().lower().split(';')[0] for line in f if line.strip())
       except FileNotFoundError: pass
           
   return {
       "novelty": novelty,
       "personal_wordlist": personal_wordlist,
       "exclude_patterns": [re.compile(p, re.I) for p in args.exclude.split(',') if p.strip()] if args.exclude else []
   }

def process_corpus(args: argparse.Namespace, resources: Dict[str, Any], corpus_hash: str) -> Dict[str, Any]:
   """Reads the corpus, tokenizes, and counts n-grams with periodic checkpointing."""
   checkpoint_dir = pathlib.Path(args.checkpoint_dir)
   
   chunk_files = sorted(checkpoint_dir.glob(f"corpus_{corpus_hash}_chunk_*.pkl"))
   start_chunk = len(chunk_files)
   
   if start_chunk > 0:
       print(f"[info] Resuming corpus scan from chunk {start_chunk}", file=sys.stderr)

   stream = stream_corpus(args)
   pbar = tqdm(total=args.max_docs, desc=f"Streaming {args.corpus}", initial=start_chunk * args.checkpoint_interval)
   
   for _ in range(start_chunk * args.checkpoint_interval):
       try:
           next(stream)
       except StopIteration:
           break
   
   chunk_data = {
       'docs_processed': 0, 'total_tokens': 0, 'unigram': Counter(),
       'doc_frequency_counter': Counter(),
   }
   for n in range(2, 6):
       chunk_data[f'counter_{n}'] = Counter()
       chunk_data[f'total_{n}'] = 0
   
   docs_in_chunk = 0
   
   total_tokens_seen = 0
   for doc_id, raw_text in enumerate(stream, start=start_chunk * args.checkpoint_interval):
       if doc_id >= args.max_docs: break
       
       text = normalize(raw_text)
       if not is_reasonable(text): continue
       if args.langid_check and langid.classify(text[:1000])[0] != 'en': continue
       
       lowers = [t for t in tokenize(text, args.min_len, args.max_len) 
                 if not any(p.search(t) for p in resources['exclude_patterns'])]
       if not lowers: continue
        # Enforce global token cap
       if args.max_tokens and (total_tokens_seen + len(lowers)) > args.max_tokens:
           break


       chunk_data['docs_processed'] += 1
       chunk_data['total_tokens'] += len(lowers)
       total_tokens_seen += len(lowers)
  
       chunk_data['unigram'].update(lowers)


       seen_in_doc = set()
       gram_configs = {2: args.keep_bigrams, 3: args.keep_trigrams, 4: args.keep_fourgrams, 5: args.keep_fivegrams}
       for n, keep in gram_configs.items():
           if keep and len(lowers) >= n:
               for gram in ngrams(lowers, n):
                   chunk_data[f'counter_{n}'][gram] += 1
                   chunk_data[f'total_{n}'] += 1
                   seen_in_doc.add((n, gram))
       
       for item in seen_in_doc:
           chunk_data['doc_frequency_counter'][item] += 1
       
       docs_in_chunk += 1
       pbar.update(1)

       if docs_in_chunk >= args.checkpoint_interval:
           chunk_num = (doc_id // args.checkpoint_interval)
           chunk_file = checkpoint_dir / f"corpus_{corpus_hash}_chunk_{chunk_num}.pkl"
           print(f"\n[info] Saving checkpoint for chunk {chunk_num} to {chunk_file}", file=sys.stderr)
           with open(chunk_file, 'wb') as f:
               pickle.dump(chunk_data, f)
           
           chunk_data = {'docs_processed': 0, 'total_tokens': 0, 'unigram': Counter(), 'doc_frequency_counter': Counter()}
           for n in range(2, 6):
               chunk_data[f'counter_{n}'] = Counter()
               chunk_data[f'total_{n}'] = 0
           docs_in_chunk = 0

   pbar.close()
   
   if docs_in_chunk > 0:
       chunk_num = (doc_id // args.checkpoint_interval)
       chunk_file = checkpoint_dir / f"corpus_{corpus_hash}_chunk_{chunk_num}.pkl"
       print(f"\n[info] Saving final chunk {chunk_num} to {chunk_file}", file=sys.stderr)
       with open(chunk_file, 'wb') as f:
           pickle.dump(chunk_data, f)

   return merge_checkpoints(corpus_hash, checkpoint_dir)


def merge_checkpoints(corpus_hash: str, checkpoint_dir: pathlib.Path) -> Dict[str, Any]:
   """Merges all chunk checkpoints into a single final corpus data object."""
   chunk_files = sorted(checkpoint_dir.glob(f"corpus_{corpus_hash}_chunk_*.pkl"))
   if not chunk_files:
       print("[warn] No corpus chunks found to merge.", file=sys.stderr)
       return {}

   print(f"[info] Merging {len(chunk_files)} corpus chunks...", file=sys.stderr)
   
   final_data = {
       "counters": {n: Counter() for n in range(2, 6)}, "unigram": Counter(),
       "doc_frequency_counter": Counter(), "total_tokens": 0, "docs_processed": 0,
       "totals": defaultdict(int)
   }

   for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
       with open(chunk_file, 'rb') as f:
           chunk_data = pickle.load(f)
       
       final_data['total_tokens'] += chunk_data.get('total_tokens', 0)
       final_data['docs_processed'] += chunk_data.get('docs_processed', 0)
       final_data['unigram'].update(chunk_data.get('unigram', {}))
       final_data['doc_frequency_counter'].update(chunk_data.get('doc_frequency_counter', {}))
       
       for n in range(2, 6):
           final_data['counters'][n].update(chunk_data.get(f'counter_{n}', {}))
           final_data['totals'][n] += chunk_data.get(f'total_{n}', 0)

   for chunk_file in chunk_files:
       os.remove(chunk_file)
       
   return final_data


def _merge_duckdb(corpus_hash: str, checkpoint_dir: pathlib.Path) -> Dict[str, Any]:
   # DuckDB build compatibility: some environments do not support MERGE yet.
   # We implement UPSERT as: UPDATE ... FROM tmp; then INSERT missing rows.
   chunk_files = sorted(checkpoint_dir.glob(f"corpus_{corpus_hash}_chunk_*.pkl"))
   if not chunk_files:
       return {}
   db_path = checkpoint_dir / f"corpus_{corpus_hash}.duckdb"
   if os.path.exists(db_path):
       os.remove(db_path)
   con = duckdb.connect(str(db_path))
   con.execute("""

       CREATE TABLE meta (k TEXT, v TEXT);

       CREATE TABLE unigram (token TEXT, cnt BIGINT);

       CREATE TABLE bigram (a TEXT, b TEXT, cnt BIGINT);

       CREATE TABLE trigram (t1 TEXT, t2 TEXT, t3 TEXT, cnt BIGINT);

       CREATE TABLE fourgram (t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, cnt BIGINT);

       CREATE TABLE fivegram (t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, t5 TEXT, cnt BIGINT);

       CREATE TABLE df2 (a TEXT, b TEXT, cnt BIGINT);

       CREATE TABLE df3 (t1 TEXT, t2 TEXT, t3 TEXT, cnt BIGINT);

       CREATE TABLE df4 (t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, cnt BIGINT);

       CREATE TABLE df5 (t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, t5 TEXT, cnt BIGINT);

   """)
   totals = defaultdict(int); total_tokens = 0; docs_processed = 0

   def upsert_simple(table: str, key_cols: list, tmp_name: str):
       # UPDATE existing
       set_expr = "t.cnt = t.cnt + s.cnt"
       join_cond = " AND ".join([f"t.{c} = s.{c}" for c in key_cols])
       con.execute(f"UPDATE {table} t SET {set_expr} FROM {tmp_name} s WHERE {join_cond}")
       # INSERT missing
       cols = ", ".join(key_cols + ["cnt"])
       sel_cols = ", ".join([f"s.{c}" for c in key_cols] + ["s.cnt"])
       null_check = " AND ".join([f"t.{c} IS NULL" for c in key_cols])
       left_join = " AND ".join([f"t.{c} = s.{c}" for c in key_cols])
       con.execute(f"INSERT INTO {table} ({cols}) SELECT {sel_cols} FROM {tmp_name} s LEFT JOIN {table} t ON {left_join} WHERE {null_check}")

   for chunk_file in chunk_files:
       with open(chunk_file, 'rb') as f:
           ch = pickle.load(f)
       total_tokens += int(ch.get('total_tokens', 0))
       docs_processed += int(ch.get('docs_processed', 0))

       # Unigram
       uni = list(ch.get('unigram', {}).items())
       if uni:
           con.execute("CREATE TEMP TABLE tu(token TEXT, cnt BIGINT)")
           con.executemany("INSERT INTO tu VALUES(?,?)", uni)
           upsert_simple("unigram", ["token"], "tu")
           con.execute("DROP TABLE tu")

       # Doc freq
       df = ch.get('doc_frequency_counter', {})
       if df:
           r2=[(*t, c) for (n, t), c in df.items() if n == 2]
           r3=[(*t, c) for (n, t), c in df.items() if n == 3]
           r4=[(*t, c) for (n, t), c in df.items() if n == 4]
           r5=[(*t, c) for (n, t), c in df.items() if n == 5]
           if r2:
               con.execute("CREATE TEMP TABLE t2(a TEXT, b TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO t2 VALUES(?,?,?)", r2)
               upsert_simple("df2", ["a","b"], "t2"); con.execute("DROP TABLE t2")
           if r3:
               con.execute("CREATE TEMP TABLE t3(t1 TEXT, t2 TEXT, t3 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO t3 VALUES(?,?,?,?)", r3)
               upsert_simple("df3", ["t1","t2","t3"], "t3"); con.execute("DROP TABLE t3")
           if r4:
               con.execute("CREATE TEMP TABLE t4(t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO t4 VALUES(?,?,?,?,?)", r4)
               upsert_simple("df4", ["t1","t2","t3","t4"], "t4"); con.execute("DROP TABLE t4")
           if r5:
               con.execute("CREATE TEMP TABLE t5(t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, t5 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO t5 VALUES(?,?,?,?,?,?)", r5)
               upsert_simple("df5", ["t1","t2","t3","t4","t5"], "t5"); con.execute("DROP TABLE t5")

       # n-grams
       for n in range(2, 6):
           totals[n] += int(ch.get(f'total_{n}', 0))
           cnt = ch.get(f'counter_{n}', {})
           if not cnt:
               continue
           rows = [(*k, v) for k, v in cnt.items()]
           if n == 2:
               con.execute("CREATE TEMP TABLE tb(a TEXT, b TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO tb VALUES(?,?,?)", rows)
               upsert_simple("bigram", ["a","b"], "tb"); con.execute("DROP TABLE tb")
           elif n == 3:
               con.execute("CREATE TEMP TABLE tt(t1 TEXT, t2 TEXT, t3 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO tt VALUES(?,?,?,?)", rows)
               upsert_simple("trigram", ["t1","t2","t3"], "tt"); con.execute("DROP TABLE tt")
           elif n == 4:
               con.execute("CREATE TEMP TABLE tf(t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO tf VALUES(?,?,?,?,?)", rows)
               upsert_simple("fourgram", ["t1","t2","t3","t4"], "tf"); con.execute("DROP TABLE tf")
           else:
               con.execute("CREATE TEMP TABLE tv(t1 TEXT, t2 TEXT, t3 TEXT, t4 TEXT, t5 TEXT, cnt BIGINT)")
               con.executemany("INSERT INTO tv VALUES(?,?,?,?,?,?)", rows)
               upsert_simple("fivegram", ["t1","t2","t3","t4","t5"], "tv"); con.execute("DROP TABLE tv")
       del ch

   # Cleanup chunks
   for cf in chunk_files:
       try: os.remove(cf)
       except OSError: pass

   con.execute("DELETE FROM meta")
   meta_rows = [("total_tokens", str(total_tokens)), ("docs_processed", str(docs_processed))] + [(f"total_{n}", str(totals[n])) for n in range(2,6)]
   con.executemany("INSERT INTO meta VALUES(?,?)", meta_rows)
   return {
       "db_path": str(db_path),
       "total_tokens": total_tokens,
       "docs_processed": docs_processed,
       "totals": defaultdict(int, {n: totals[n] for n in range(2,6)}),
       "unigram": DuckCounter(str(db_path), "unigram", 1),
       "counters": {2: DuckCounter(str(db_path), "bigram", 2),
                    3: DuckCounter(str(db_path), "trigram", 3),
                    4: DuckCounter(str(db_path), "fourgram", 4),
                    5: DuckCounter(str(db_path), "fivegram", 5)},
       "doc_frequency_counter": DuckDocFreq(str(db_path)),
   }


def save_corpus_manifest(manifest_path: pathlib.Path, backend: str, db_path: str, totals: Dict[int,int], total_tokens: int, docs_processed: int):
    data = {
        "backend": backend,
        "db_path": str(db_path),
        "totals": {int(k): int(v) for k, v in totals.items()},
        "total_tokens": int(total_tokens),
        "docs_processed": int(docs_processed),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_corpus_manifest(manifest_path: pathlib.Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    backend = m.get("backend", "sqlite-fast")
    db_path = m["db_path"]
    totals = defaultdict(int, {int(k): int(v) for k, v in m.get("totals", {}).items()})
    total_tokens = int(m.get("total_tokens", 0))
    docs_processed = int(m.get("docs_processed", 0))
    # Reconstruct proxy objects based on backend
    if backend == "duckdb" and 'DuckCounter' in globals():
        return {
            "db_path": db_path,
            "total_tokens": total_tokens,
            "docs_processed": docs_processed,
            "totals": totals,
            "unigram": DuckCounter(db_path, "unigram", 1),
            "counters": {2: DuckCounter(db_path, "bigram", 2),
                         3: DuckCounter(db_path, "trigram", 3),
                         4: DuckCounter(db_path, "fourgram", 4),
                         5: DuckCounter(db_path, "fivegram", 5)},
            "doc_frequency_counter": DuckDocFreq(db_path),
        }
    if backend == "lmdb" and 'LmdbCounter' in globals():
        return {
            "db_path": db_path,
            "total_tokens": total_tokens,
            "docs_processed": docs_processed,
            "totals": totals,
            "unigram": LmdbCounter(db_path, 1),
            "counters": {2: LmdbCounter(db_path, 2),
                         3: LmdbCounter(db_path, 3),
                         4: LmdbCounter(db_path, 4),
                         5: LmdbCounter(db_path, 5)},
            "doc_frequency_counter": LmdbDocFreq(db_path),
        }
    # default: sqlite proxies
    return {
        "db_path": db_path,
        "total_tokens": total_tokens,
        "docs_processed": docs_processed,
        "totals": totals,
        "unigram": SqlCounter(db_path, "unigram", 1),
        "counters": {2: SqlCounter(db_path, "bigram", 2),
                     3: SqlCounter(db_path, "trigram", 3),
                     4: SqlCounter(db_path, "fourgram", 4),
                     5: SqlCounter(db_path, "fivegram", 5)},
        "doc_frequency_counter": SqlDocFreq(db_path),
    }


def calculate_final_score(base_score: float, scrabble_score: float, semantic_score: float, phonetic_score: float, args: argparse.Namespace) -> float:
   """Combines all score components into a final weighted score."""
   base_weight = 1.0 - (args.semantic_weight + args.phonetic_weight)
   return (base_score * base_weight) + \
          (semantic_score * args.semantic_weight) + (phonetic_score * args.phonetic_weight) + (scrabble_score * args.scrabble_weight)

worker_globals = {}
def init_worker(args, resources, corpus_data):
   """Initializes resources for each worker process, especially the spaCy model."""
   worker_globals['args'] = args
   worker_globals['resources'] = resources
   worker_globals['corpus_data'] = corpus_data
   
   nlp_model = None
   if args.pos_filter or args.semantic_weight > 0:
       model_name = "en_core_web_md" if args.semantic_weight > 0 else "en_core_web_sm"
       try:
           nlp_model = spacy.load(model_name, disable=["parser", "ner"])
       except OSError:
           print(f"[warn] Worker PID {os.getpid()} could not load spaCy model '{model_name}'.", file=sys.stderr)
   worker_globals['nlp_model'] = nlp_model

def process_chunk(chunk_with_n: Tuple) -> Tuple[int, List[Tuple]]:
   """Processes a chunk of n-grams in a worker process."""
   chunk, n = chunk_with_n
   args = worker_globals['args']
   resources = worker_globals['resources']
   corpus_data = worker_globals['corpus_data']
   nlp_model = worker_globals['nlp_model']
   scored_phrases = []

   if nlp_model and chunk:
        docs_iter = nlp_model.pipe((" ".join(tup) for tup, _ in chunk), batch_size=256)
   else:
        docs_iter = (None for _ in chunk)

   for (tup, cnt), doc in zip(chunk, docs_iter):
       phrase = " ".join(tup)
       letters = sum(len(t) for t in tup)
       
       if not (args.min_phrase_chars <= letters <= args.max_phrase_chars): continue
       if not accept_tokens(tup, args): continue
       if resources["personal_wordlist"]:
            norms = {phrase, phrase.replace(" ", ""), phrase.replace(" ", "_")}
            if any(p in resources["personal_wordlist"] for p in norms):
                continue

       z = zipf(cnt, corpus_data["totals"][n])
       if z < args.min_zipf_phrase: continue

       df = corpus_data["doc_frequency_counter"].get((n, tup), 0)
       if df < args.min_docs: continue

       assoc = assoc_bridge(n, tup, args.score, corpus_data["unigram"], corpus_data["counters"][2], corpus_data["total_tokens"], corpus_data["totals"][2])
       
       if (args.score == 'llr' and assoc < args.llr_min) or (args.score == 'npmi' and assoc < args.pmi_min): continue
       if args.pos_filter and not is_grammatically_interesting(doc): continue

       normalized_assoc = math.log1p(assoc) / 5.0 if args.score == 'llr' else assoc
       base_score = (0.75 * normalized_assoc) + (0.05 * z) + (0.2 * min(1.0, df / 50.0))
       semantic_score = score_semantic_surprise(doc)
       phonetic_score = score_phonetic_appeal(tup)
       
       final_score = calculate_final_score(base_score, semantic_score, phonetic_score, score_scrabble(tup), args)
       scored_phrases.append((phrase, cnt, z, round(assoc, 3), df, round(final_score, 3), letters))
       
   return len(chunk), scored_phrases

def filter_and_score_phrases(args: argparse.Namespace, resources: Dict, corpus_data: Dict, filter_checkpoint_hash: str) -> Dict[str, List[Tuple]]:
   """Uses a multiprocessing pool to filter and score all candidate n-grams."""
   scored_phrases_by_type = {}
   gram_map = {2: "bigrams", 3: "trigrams", 4: "fourgrams", 5: "fivegrams"}
   checkpoint_dir = pathlib.Path(args.checkpoint_dir)
   
   num_workers = mp.cpu_count() if args.workers == -1 else args.workers
   print(f"[info] Using {num_workers} worker processes for scoring.", file=sys.stderr)

   for n, kind_name in gram_map.items():
       filter_checkpoint_file = checkpoint_dir / f"{filter_checkpoint_hash}_{kind_name}.pkl"
       if filter_checkpoint_file.exists():
           print(f"[info] Found scored checkpoint for {kind_name}. Loading from {filter_checkpoint_file}", file=sys.stderr)
           with open(filter_checkpoint_file, 'rb') as f:
               scored_phrases_by_type[kind_name] = pickle.load(f)
           continue

       counter = corpus_data["counters"][n]
       if not counter: continue
       total_candidates = len(counter)
       chunk_size = max(1, min(args.worker_chunk_size, max(1, total_candidates // (num_workers * 4))))
       def _iter_chunks(mapping, size):
            it = iter(mapping.items())
            while True:
                batch = list(itertools.islice(it, size))
                if not batch:
                    break
                yield batch
       tasks = ((batch, n) for batch in _iter_chunks(counter, chunk_size))
       topk = args.top_k_per_type if args.top_k_per_type and args.top_k_per_type > 0 else None
       if topk:
            import heapq
            heap = []  # (key, idx, row)
            idx_counter = 0
       else:
            scored_phrases = []
       with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(args, resources, corpus_data)) as pool:
            with tqdm(total=total_candidates, desc=f"Filtering {kind_name}") as pbar:
                for processed_count, result_chunk in pool.imap_unordered(process_chunk, tasks):
                    if topk:
                        for row in result_chunk:
                            key = (row[5], row[3], row[2])
                            if len(heap) < topk:
                                heapq.heappush(heap, (key, idx_counter, row)); idx_counter += 1
                            else:
                                if key > heap[0][0]:
                                    heapq.heapreplace(heap, (key, idx_counter, row)); idx_counter += 1
                    else:
                        scored_phrases.extend(result_chunk)
                    pbar.update(processed_count)

       if topk:
            scored_phrases = [t[-1] for t in sorted(heap, key=lambda t: t[0], reverse=True)]
       else:
            scored_phrases.sort(key=lambda r: (r[5], r[3], r[2]), reverse=True)
       scored_phrases_by_type[kind_name] = scored_phrases
       print(f"[info] Saving scored checkpoint for {kind_name} to {filter_checkpoint_file}", file=sys.stderr)
       with open(filter_checkpoint_file, 'wb') as f:
           pickle.dump(scored_phrases, f)
       
   return scored_phrases_by_type

def write_output_files(args: argparse.Namespace, scored_phrases_by_type: Dict[str, List], corpus_data: Dict[str, Any]):
   """Writes all CSV and JSON output files."""
   outbase = pathlib.Path(f"{args.out_prefix}_{args.corpus}")
   outbase.parent.mkdir(parents=True, exist_ok=True)
   
   all_novel_phrases = []
   header = ["phrase", "count", "zipf", "assoc", "doc_freq", "score", "letters"]

   for kind_name, rows in scored_phrases_by_type.items():
       with open(outbase.with_suffix(f".{kind_name}.csv"), "w", newline="", encoding="utf-8") as f:
           writer = csv.writer(f)
           writer.writerow(header)
           writer.writerows(rows)
       if args.summary_file:
           all_novel_phrases.extend([(kind_name,) + row for row in rows])

   if args.summary_file and all_novel_phrases:
       print(f"[info] Writing {len(all_novel_phrases)} novel phrases to summary file: {args.summary_file}", file=sys.stderr)
       all_novel_phrases.sort(key=lambda r: r[6], reverse=True)
       with open(args.summary_file, "w", newline="", encoding="utf-8") as f:
           writer = csv.writer(f)
           writer.writerow(["type"] + header)
           writer.writerows(all_novel_phrases)
           
   meta = {
       "docs_processed": corpus_data["docs_processed"], "total_tokens": corpus_data["total_tokens"],
       "totals": {str(k): int(v) for k, v in corpus_data["totals"].items()}, "args": vars(args)
   }
   with open(outbase.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
       json.dump(meta, f, indent=2, default=str)


def main():
   """Main function to orchestrate the phrase extraction pipeline."""
   args = setup_args()
   
   checkpoint_dir = pathlib.Path(args.checkpoint_dir)
   checkpoint_dir.mkdir(parents=True, exist_ok=True)
   
   corpus_settings = f"{args.corpus}-{args.hf_config}-{args.split}-{args.max_docs}-{args.max_tokens}"
   corpus_hash = hashlib.md5(corpus_settings.encode()).hexdigest()
   corpus_checkpoint_file = checkpoint_dir / f"corpus_{corpus_hash}.pkl"

   corpus_data = None
   if corpus_checkpoint_file.exists():
       print(f"[info] Found final corpus checkpoint. Loading from {corpus_checkpoint_file}", file=sys.stderr)
       with open(corpus_checkpoint_file, 'rb') as f:
           corpus_data = pickle.load(f)
   else:
       resources = load_resources(args)
       corpus_data = process_corpus(args, resources, corpus_hash)
       print(f"[info] Corpus scan complete. Saving final checkpoint to {corpus_checkpoint_file}", file=sys.stderr)
       with open(corpus_checkpoint_file, 'wb') as f:
           pickle.dump(corpus_data, f)

   if not corpus_data:
       sys.exit("ERROR: Corpus data is empty. Something went wrong during the scan or merge.")

   resources = load_resources(args)
   
   filter_settings_keys = ['score','min_zipf_phrase','pmi_min','llr_min','min_docs','no_edge_stops','min_content_ratio','pos_filter','semantic_weight','phonetic_weight','scrabble_weight','min_avg_scrabble']
   filter_settings = {k: vars(args)[k] for k in filter_settings_keys}
   filter_settings_str = json.dumps(filter_settings, sort_keys=True)
   filter_hash = hashlib.md5(filter_settings_str.encode()).hexdigest()
   
   filter_checkpoint_hash = f"scored_{corpus_hash}_{filter_hash}"

   scored_phrases = filter_and_score_phrases(args, resources, corpus_data, filter_checkpoint_hash)
   write_output_files(args, scored_phrases, corpus_data)
   print("[info] Processing complete.", file=sys.stderr)

if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\nInterrupted by user.", file=sys.stderr)
       sys.exit(130)
