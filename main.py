import os
import fnmatch

import pipeline

from utils import read

WINDOW_NAME = "CBIR"
DIRNAME = "DATA"


def app():
    db = []
    for f in os.listdir(DIRNAME):
        if fnmatch.fnmatch(f, '*.jpg'):
            db.append(read(DIRNAME, f))

    db = pipeline.preprocess(db[0:2])
    print(db[0].shape)


if __name__ == '__main__':
    app()
