import fnmatch
import os

import numpy as np
from sklearn.neighbors import KDTree

from pipeline import Pipeline
from utils import read


class Database:
    def __init__(self, dirname, dbname):
        self.dirname = dirname
        self.dbname = dbname
        self.db = {}
        self.pipeline = Pipeline(0)
        self.tree = None
        self.load()

    def load(self):
        if os.path.isfile(self.dbname + ".npz"):
            self.db = np.load(self.dbname + ".npz", allow_pickle=True)
            X = np.array(np.zeros(shape=(len(self.db.keys()) - 1, 16 * 16 * 3)))
            for idx, v in enumerate(self.db.values()):
                try:
                    X[idx] = v[2].flatten().reshape(1, -1)
                except IndexError:
                    continue
            self.tree = KDTree(X, leaf_size=2)
        else:
            self.generate()

    def generate(self):
        for n, f in enumerate(sorted(os.listdir(self.dirname))):
            if fnmatch.fnmatch(f, '*.jpg'):
                feature_vector = self.pipeline.process(read(self.dirname, f))
                self.db[f[0:f.index('.')]] = feature_vector

        np.savez(self.dbname, self.db, **self.db)

    def open(self):
        return self.db
