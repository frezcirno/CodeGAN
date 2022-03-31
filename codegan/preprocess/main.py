import argparse
from . import re_preprocess, cb_preprocess
from . import file_utils

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str)
parser.add_argument("-m", "--method", choices=['codegan', 'codebert'], type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)

args = parser.parse_args()

df = file_utils.read_jsonl(args.dataset_path)
df.info()

if args.method == 'codegan':
    df_processed = re_preprocess.preprocess(df)
else:
    df_processed = cb_preprocess.preprocess(df)

df_processed.info()

if not args.output.endswith('.parquet'):
    args.output += '.parquet'

df_processed.to_parquet(args.output)
