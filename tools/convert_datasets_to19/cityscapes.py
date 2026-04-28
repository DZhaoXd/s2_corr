# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_json_to_label(json_file, gt_dir, out_gt_dir):
    try:
        from cityscapesscripts.preparation.json2labelImg import json2labelImg
    except ImportError as exc:
        raise ImportError(
            "cityscapesscripts is required for Cityscapes polygon conversion. "
            "Install it with `pip install cityscapesscripts`."
        ) from exc

    rel_file = Path(json_file).relative_to(gt_dir)
    label_rel = Path(str(rel_file).replace('_polygons.json',
                                           '_labelTrainIds.png'))
    label_file = Path(out_gt_dir) / label_rel
    label_file.parent.mkdir(parents=True, exist_ok=True)
    json2labelImg(json_file, str(label_file), 'trainIds')

    if 'train' in label_rel.parts:
        pil_label = Image.open(str(label_file))
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = str(label_file)
        return sample_class_stats
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('--gt19-dir', default='gtFine_19', type=str)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val', 'test'],
        help='Cityscapes splits to convert, e.g. train val.')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [dict(e) for e in sample_class_stats
                          if e is not None]
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


def resolve_cityscapes_root(cityscapes_path):
    """Resolve common Cityscapes layouts to the directory with gtFine."""
    path = Path(cityscapes_path)
    candidates = [
        path,
        path / 'Cityscapes',
        path / 'cityscapes' / 'Cityscapes',
        path.parent / 'cityscape',
        path.parent / 'cityscape' / 'cityscapes' / 'Cityscapes',
    ]
    for candidate in candidates:
        if (candidate / 'gtFine').is_dir():
            return candidate
    return path


def main():
    args = parse_args()
    cityscapes_path = resolve_cityscapes_root(args.cityscapes_path)
    out_dir = Path(args.out_dir) if args.out_dir else cityscapes_path
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = cityscapes_path / args.gt_dir
    out_gt_dir = out_dir / args.gt19_dir
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    poly_files = []
    for split in args.splits:
        split_dir = gt_dir / split
        if split_dir.is_dir():
            poly_files.extend(str(p) for p in split_dir.rglob('*_polygons.json'))
    if not poly_files:
        raise FileNotFoundError(
            f'No Cityscapes polygon JSON files found in {gt_dir} '
            f'for splits: {args.splits}')

    only_postprocessing = False
    worker = partial(convert_json_to_label, gt_dir=gt_dir,
                     out_gt_dir=out_gt_dir)
    if not only_postprocessing:
        sample_class_stats = track_progress(worker, poly_files, args.nproc)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

    for split in args.splits:
        filenames = []
        split_dir = gt_dir / split
        if not split_dir.is_dir():
            continue
        for poly in split_dir.rglob('*_polygons.json'):
            filenames.append(
                str(poly.relative_to(split_dir)).replace(
                    '_gtFine_polygons.json', ''))
        with open(out_dir / f'{split}.txt', 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
