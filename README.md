# Content Based Image Retrieval with Wavelet features

## Installation

Run `main.py` with the following arguments to query an image using the base CBIR system. All parameters are optional. By
default, the first image in the University of Washington dataset (`./data`) is set as the query image.

The best 10 matches are retrieved by the system, displayed and written to an automatically created (`./DIRNAME_out`)
folder. The folder is emptied for each result-set.

```
python3 -m pip install --user virtualenv
python3 -m venv venv
source ./venv/bin/activate
pip3 install -Ur requirements.txt
```

## Image Query

```
python3 main.py --query_dirname query --query test.jpg 
```

Run `main.py` with the following arguments to change the number of returned matches, or the second-stage comparison
distance measure threshold.

```
python3 main.py --query_dirname query --query test.jpg --matches 20 --threshold 20000
```

Run `main.py` with the `--delay` flag to increase the viewing time of the query image from the default three seconds.
Note that the delay is specified in milliseconds.

```
python3 main.py --query_dirname query --query test.jpg --delay 5000
```

## System Optimisations

Run `main.py` with the `--pca` flag to use PCA to reduce the dimensionality of the feature vectors. The feature vector
database, if it does not exist, will be saved to disk as `DBNAME_pca.npz`.

```
python3 main.py --query_dirname query --query test.jpg --pca
```

Run `main.py` with the `--kdtree` flag to store the $16 \times 16 \times 3$ feature vectors in a K-d tree. This flag can
be combined \texttt{--pca}. In this configuration, the K-d tree stores the reduced feature vectors.

```
python3 main.py --query_dirname query --query test.jpg --kdtree
```

## Distance Measure Weight Adjustment

Run `main.py` with the `--horizontal` flag to adjust the weight w_{12} by 20%. This elevates the horizontal image detail
in the comparison procedure. The default adjustment factor is 1.5, or 50%

```
python3 main.py --query_dirname query --query test.jpg --horizontal --scale 1.2
```

Including the `--vertical`, `--diagonal` or `--intensity` flag will similarly adjust the corresponding weight. All flags
and combinations thereof can be specified.

## Help

Run `main.py` with the `--help` flag to view the Help file.

```
python3 main.py --help
```

```
usage: main.py [-h] [--dbname DBNAME] [--dirname DIRNAME] [--query_dirname QUERY_DIRNAME] 
[--query QUERY] [--matches MATCHES] [--delay DELAY] [--threshold THRESHOLD] [--vertical] 
[--horizontal] [--diagonal] [--intensity] [--scale SCALE] [--pca] [--kdtree]

Content Based Image Retrieval with Wavelet features

optional arguments:
  -h, --help            show this help message and exit
  --dbname DBNAME       Database name
  --dirname DIRNAME     Image database directory name
  --query_dirname QUERY_DIRNAME
                        Query image directory name
  --query QUERY         Query image filename
  --matches MATCHES     Number of matches to return
  --delay DELAY         OpenCV window millisecond delay
  --threshold THRESHOLD
                        Euclidean distance threshold
  --vertical            Emphasise vertical image detail
  --horizontal          Emphasise horizontal image detail
  --diagonal            Emphasise diagonal image detail
  --intensity           Emphasise intensity variation
  --scale SCALE         Weight scaling factor
  --pca                 Perform feature dimensionality reduction
  --kdtree              Store features in K-d tree

```