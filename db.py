import fnmatch
import os

import numpy as np

import pipeline
from utils import read


class Database:
    def __init__(self, dirname, dbname):
        self.dirname = dirname
        self.dbname = dbname
        self.db = {}
        self.load()

    def load(self):
        if os.path.isfile(self.dbname):
            self.db = np.load(self.dbname)
        else:
            self.generate()

    def generate(self):
        for n, f in enumerate(sorted(os.listdir(self.dirname))):
            if fnmatch.fnmatch(f, '*.jpg'):
                feature_vector = pipeline.pipe(read(self.dirname, f))
                self.db[f[0:f.index('.')]] = feature_vector

        np.savez(self.dbname, self.db, **self.db)

    def open(self):
        return self.db
