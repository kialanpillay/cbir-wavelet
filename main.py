import argparse
import os
import time

import cv2
from art import tprint

import utils
from db import Database
from pipeline import Pipeline

WINDOW_NAME = "CBIR"
VERSION = 1.0


def app():
    tprint("CBIR", "rounded")
    print("CBIR")
    print("Content Based Image Retrieval with Wavelet features")
    print("Version {}".format(VERSION))
    print("© Kialan Pillay 2021")
    print("-" * 50)
    print()

    query_img = None
    if os.path.isfile(os.path.join(args.dirname, args.query)):
        query_img = utils.read(args.dirname, args.query)
        cv2.imshow(WINDOW_NAME, query_img)
        cv2.waitKey(args.delay)
        cv2.destroyAllWindows()
    else:
        print("CBIR - {} not found...".format(args.query))
        exit(1)

    print("Building pipeline...")
    t1 = time.time()
    pipeline = Pipeline(args.threshold, args.vertical, args.horizontal, args.diagonal, args.color, args.scale)
    t2 = time.time()
    system_time(t1, t2)

    print("Forming feature vector...")
    t1 = time.time()
    Q = pipeline.pipe(query_img)
    t2 = time.time()
    system_time(t1, t2)

    print("Loading feature vector database...")
    t1 = time.time()
    db = Database(args.dirname, args.dbname).open()
    t2 = time.time()
    system_time(t1, t2)

    if args.test:
        matches = []
        print("Performing three-stage comparison...")
        t1 = time.time()
        for k, v in db.items():
            if k == 'arr_0':
                continue
            dist = pipeline.filter_(Q, v)
            matches.append((k, dist))
        t2 = time.time()
        system_time(t1, t2)

        print("Displaying final query results...\n")
        print("{} Query Results - Best {} Matches ".format(args.query[0:args.query.index('.')], args.matches))
        print("-" * 50)
        print("{0:<6} {1:<20} {2:<6}".format("Rank", "Image", "Dist"))

        if not os.path.isdir(output_dir(args.dirname)):
            os.mkdir(output_dir(args.dirname))
        else:
            [os.remove(os.path.join(output_dir(args.dirname), f)) for f in os.listdir(output_dir(args.dirname))
             if f.endswith(".jpg")]

        for n, (file, dist) in enumerate(sorted(matches, key=lambda x: x[1])):
            print("{0:<6} {1:<25} {2:<05.2f}".format(n + 1, file, dist))
            utils.write(args.dirname + "_out", file)
            if n > (args.matches - 1):
                break


def output_dir(dirname):
    return dirname + "_out"


def system_time(t1, t2):
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="main.py", description="Content Based Image Retrieval with Wavelet features")
    parser.add_argument('--dbname', type=str, default='db', help="Database name")
    parser.add_argument('--dirname', type=str, default='data', help="Image database directory name")
    parser.add_argument('--query', type=str, default='arborgreens01.jpg', help="Query image filename")
    parser.add_argument('--matches', type=int, default=10, help="Number of matches to return")
    parser.add_argument('--delay', type=int, default=3000, help="OpenCV window millisecond delay")
    parser.add_argument('--threshold', type=int, default=30000, help="Euclidean distance threshold")
    parser.add_argument('--test', action='store_true', help="Evaluation mode")
    parser.add_argument('--vertical', action='store_true', help="Emphasise vertical image detail")
    parser.add_argument('--horizontal', action='store_true', help="Emphasise horizontal image detail")
    parser.add_argument('--diagonal', action='store_true', help="Emphasise diagonal image detail")
    parser.add_argument('--color', action='store_true', help="Emphasise color variation")
    parser.add_argument('--scale', type=float, default=1.5, help="Weight scaling factor")
    parser.add_argument('--pca', action='store_true', help="Perform feature dimensionality reduction")
    args = parser.parse_args()
    app()
