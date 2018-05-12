# Reddit Clusters

Understanding hateful subreddits through text clustering.

## Directory Structure

```
.
├── clustering
│   ├── results
│   │   ├── CringeAnarchy.txt
│   │   ├── The_Donald.txt
│   │   └── TheRedPill.txt
│   ├── run_nmfs.sh
│   └── tfidf_nmf.py
├── data
│   ├── bigquery
│   │   └── 2017
│   │       └── 11-12
│   │           └── download_data.sh
│   └── stoplist.txt
├── LICENSE
├── README.md
└── wordclouds
    ├── blackdisc.jpeg
    ├── images
    │   ├── CringeAnarchy
    │   │   ├── 0.06%.png
    │   │   ├── 0.12%.png
    │   │   ├── ...
    │   │   ├── ...
    │   ├── The_Donald
    │   │   ├── 0.36%.png
    │   │   ├── 0.51%.png
    │   │   ├── ...
    │   │   ├── ...
    │   └── TheRedPill
    │       ├── 0.56%.png
    │       ├── 0.91%.png
    │       ├── ...
    │       └── ...
    ├── make_wordclouds.ipynb
    └── OCR-A-Std-Regular.ttf
```

The `clustering` directory contains all code used to vectorize and cluster the
subreddits. `tfidf_nmf.py` is the main program, and `run_nmfs.sh` is simply a
driver script. The `results` subdirectory contains the log files when running
`tfidf_nmf.py` on `/r/TheRedPill`, `/r/The_Donald` and `/r/CringeAnarchy`.

The `data` directory contains `download_data.sh` (which downloads the Reddit
data from my Google Cloud Storage) and `stoplist.txt` (which includes
Reddit-specific words such as "moderator", "karma", etc.).

The `wordclouds` directory contains `images` (which contains png files of the
wordclouds themselves) and `make_wordclouds.ipynb` (which generates the
wordclouds). Note that 1) `blackdisc.jpeg` and `OCR-A-Std-Regular.ttf` are just
helper files to create the wordclouds, and 2) the png files are named after
their cluster importance (e.g. `0.50%.png` is a wordcloud whose cluster has an
importance of 0.50%).
