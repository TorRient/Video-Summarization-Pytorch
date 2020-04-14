#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.utils import Logger, read_json, write_json, save_checkpoint
from networks.DSN import *
from utils.rewards import compute_reward
from utils import vsum_tool
from utils.generate_dataset import Generate_Dataset
import cv2

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-i', '--input', type=str, default='', help="input video")
parser.add_argument('-o', '--output',type=str, default='./makedata/', help="output video")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
# Model options
parser.add_argument('--input-dim', type=int, default=2048, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")

parser.add_argument('-d', '--dataset', type=str, help="path to h5 dataset (required)")

parser.add_argument('--model', type=str, default='model/best_model_epoch60.pth.tar', help="path to model file")
parser.add_argument('--save-dir', type=str, default='output/', help="path to save output (default: 'output/')")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")

parser.add_argument('--save-name', default='',help="'generate video '")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")

parser.add_argument('--train-data', action='store_true', help="")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    test_keys = []

    for key in dataset.keys():
        test_keys.append(key)

    print("Load model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.model:
        print("Loading checkpoint from '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    evaluate(model, dataset, test_keys, use_gpu)
    print("Summary")
    video2summary(os.path.join(args.save_dir,'result.h5'),args.input ,args.save_dir)####

def evaluate(model, dataset, test_keys, use_gpu):
    with torch.no_grad():
        model.eval()
        fms = []

        table = [["No.", "Video", "F-score"]]
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        h5_res = h5py.File(os.path.join(args.save_dir,'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            video_name = dataset[key]['video_name'][()]
            fps = dataset[key]['fps'][()]

            sum = 0
            for i in range(len(nfps)):
                sum += nfps[i]

            machine_summary = vsum_tool.generate_summary(probs, cps, num_frames, nfps, positions)
            h5_res.create_dataset(key + '/score', data=probs)
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
            h5_res.create_dataset(key + '/video_name', data=video_name)
            h5_res.create_dataset(key + '/fps', data=fps)

    h5_res.close()

def frm2video(video_dir, summary, vid_writer):
    print('[INFO] Video Summary')
    video_capture = cv2.VideoCapture(video_dir)
    count = 0
    for idx, val in tqdm(enumerate(summary)):
        ret, frame = video_capture.read()
        if val == 1 and ret:
            frm = cv2.resize(frame, (args.width, args.height))
            vid_writer.write(frm)
        else:
            count += 1
    print('[OUTPUT] total {} frame, ignore {} frame'.format(len(summary)-count, count))
def video2summary(h5_dir,video_dir,output_dir):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    h5_res = h5py.File(h5_dir, 'r')

    for idx1 in range(len(list(h5_res.keys()))):
        key = list(h5_res.keys())[idx1]
        summary = h5_res[key]['machine_summary'][...]
        video_name = h5_res[key]['video_name'][()].split('/')[-1]
        fps = h5_res[key]['fps'][()]
        if not os.path.isdir(osp.join(output_dir, video_name)):
            os.mkdir(osp.join(output_dir, video_name))
        vid_writer = cv2.VideoWriter(
            osp.join(output_dir,video_name, args.save_name),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            fps,
            (args.width, args.height),
        )
        frm2video(video_dir, summary, vid_writer)
        vid_writer.release()
    h5_res.close()

if __name__ == '__main__':
    name_video = args.input.split('/')[-1].split('.')[0]
    args.dataset = os.path.join(args.output, name_video + '.h5')
    args.save_name = name_video + '.mp4'
    if not os.path.exists(args.dataset):
        gen = Generate_Dataset(args.input, args.dataset)
        gen.generate_dataset()
        gen.h5_file.close()
    main()
