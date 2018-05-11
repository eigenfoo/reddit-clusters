# Reddit Clusters

Understanding hateful subreddits through text clustering.

## Directory Structure

```
.
├── clustering
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
    ├── images
    └── make_wordclouds.ipynb
```

The `clustering` directory contains all code used to vectorize and cluster the
subreddits. `tfidf_nmf.py` is the main program, and `run_nmfs.sh` is simply a
driver script.

The `data` directory contains `download_data.sh` (which downloads the Reddit
data from my Google Cloud Storage) and `stoplist.txt` (which includes
Reddit-specific words such as "moderator", "karma", etc.).

The `wordclouds` directory contains `images` (which contains .png files of the
wordclouds themselves) and `make_wordclouds.ipynb` (which generates the
wordclouds).
