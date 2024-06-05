import os
import argparse
import pickle

from decord import VideoReader, cpu


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='/data/kinetics400')
parser.add_argument('--filedir', type=str, default='train2')
args = parser.parse_args()

# get entire mp4 files
labels = []
for class_idx, dir in enumerate(os.listdir(args.filedir)):
    for filename in os.listdir(os.path.join(args.filedir, dir)):
        labels.append((class_idx, os.path.join(args.filedir, dir, filename)))

print(len(labels))

# get the mp4 that is not readable
removing_paths = []
for idx, label in enumerate(labels):
    file_path = label[1]
    try:
        vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    except:
        print('no vr', file_path)
        removing_paths.append(file_path)
        continue
    print(f'[{idx} / {len(labels)}]', end='\r')

print(len(removing_paths))

# save the readable mp4
filtered_labels = [tup for tup in labels if tup[1] not in removing_paths]

print(len(filtered_labels))

with open('./labels/label_full_1.0.pickle', 'wb') as f:
    pickle.dump(filtered_labels, f)
