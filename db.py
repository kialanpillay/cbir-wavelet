import fnmatch
import os

import numpy as np
from sklearn.neighbors import KDTree

from pipeline import Pipeline
from utils import read


class Database:
    def __init__(self, dirname, dbname, pca):
        self.dirname = dirname
        self.dbname = dbname
        self.pca = pca
        self.db = {}
        self.pipeline = Pipeline(0, pca=pca)
        self.tree = None
        self.load()

    def load(self):
        if os.path.isfile(self.dbname + ".npz") and self.pca is False:
            self.db = np.load(self.dbname + ".npz", allow_pickle=True)
            X = np.array(np.zeros(shape=(len(self.db.keys()) - 1, 16 * 16 * 3)))
            for idx, v in enumerate(self.db.values()):
                try:
                    X[idx] = v[2].flatten().reshape(1, -1)
                except IndexError:
                    continue
            self.tree = KDTree(X, leaf_size=2)
        elif os.path.isfile(self.dbname + "_pca.npz") and self.pca:
            self.db = np.load(self.dbname + "_pca.npz", allow_pickle=True)
            X = np.array(np.zeros(shape=(len(self.db.keys()) - 1, 2 * 16 * 3)))
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

        if self.pca:
            file = self.dbname + "_pca"
        else:
            file = self.dbname
        np.savez(file, self.db, **self.db)

    def open(self):
        return self.db
