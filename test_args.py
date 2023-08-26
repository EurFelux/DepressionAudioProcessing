import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=3)
args = parser.parse_args()

print(args.test)