# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_to_train_id(file, gt_dir, out_label_dir):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    rel_file = Path(file).relative_to(gt_dir)
    new_file = Path(out_label_dir) / rel_file
    new_file.parent.mkdir(parents=True, exist_ok=True)
    sample_class_stats['file'] = str(new_file)
    Image.fromarray(label_copy, mode='L').save(str(new_file))
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('--label19-dir', default='labels_19', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [dict(e) for e in sample_class_stats]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def track_progress(func, files, nproc):
    if nproc > 1:
        with Pool(nproc) as pool:
            return list(tqdm(pool.imap(func, files), total=len(files)))
    return [func(file) for file in tqdm(files)]


def resolve_gta_root(gta_path):
    """Resolve common GTA5 layouts to the GTAV directory."""
    path = Path(gta_path)
    candidates = [
        path,
        path / 'GTAV',
        path / 'GTA5' / 'GTAV',
        path.parent / 'GTA5' / 'GTAV',
    ]
    for candidate in candidates:
        if (candidate / 'labels').is_dir():
            return candidate
    return path


def main():
    args = parse_args()
    gta_path = resolve_gta_root(args.gta_path)
    out_dir = Path(args.out_dir) if args.out_dir else gta_path
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = gta_path / args.gt_dir
    out_label_dir = out_dir / args.label19_dir
    out_label_dir.mkdir(parents=True, exist_ok=True)

    poly_files = [
        str(p) for p in gt_dir.rglob('*.png')
        if p.name[-5:-4].isdigit()
    ]
    poly_files = sorted(poly_files)
    if not poly_files:
        raise FileNotFoundError(f'No GTA label PNG files found in {gt_dir}')

    only_postprocessing = False
    worker = partial(
        convert_to_train_id, gt_dir=gt_dir, out_label_dir=out_label_dir)
    if not only_postprocessing:
        sample_class_stats = track_progress(worker, poly_files, args.nproc)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
