import argparse
from . import process
from utils.data_utils import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str)
parser.add_argument("-o", "--output", type=str, required=True)

args = parser.parse_args()

df = read_jsonl(args.dataset_path)
df.info()

df_processed = process.process(df)
df_processed.info()

if not args.output.endswith('.parquet'):
    args.output += '.parquet'

df_processed.to_parquet(args.output)
