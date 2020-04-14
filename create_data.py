import argparse
from utils.generate_dataset import Generate_Dataset

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--input', '--split', type=str, help="input video")
parser.add_argument('--output', type=str, default='', help="out data")

args = parser.parse_args()
if __name__ == "__main__":
    gen = Generate_Dataset(args.input, args.output)
    gen.generate_dataset()
    gen.h5_file.close()