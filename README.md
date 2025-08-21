# uncrawlr
A Python based tool for extracting n-grams from AI training corpora

**This is a pre-alpha release - some fiddling around in python will be needed!**

You will need to pip install a number of dependencies:
>datasets, ftfy, tdqm, langid, spacy, jellyfish, zstandard, pandas, itertools, huggingface_hub
>and probably more that i've forgotten

# usage

The following is a good place to start. You'll need either a local download of a corpus like openwebtext or stream one like fineweb from huggingface
Tweak the semantic and scrabble scoring to your liking

--corpus fineweb --max-docs 10000 --keep-fourgrams --keep-fivegrams --pos-filter --workers -1 --checkpoint-interval 10000 --worker-chunk-size 1500 --top-k-per-type 200000 --checkpoint-dir ./checkpoints --scrabble-weight 0.3 --min-avg-scrabble 1.4 --keep-bigrams --keep-trigrams --semantic-weight 0.2 --phonetic-weight 0.2

Once you have some .csv, use cleanup.py to process them into a text file which you'll then need to manually curate. cleanup.py contains options to filter out entries from your own wordlists, previous hits, words at start/end/blacklist etc. 


