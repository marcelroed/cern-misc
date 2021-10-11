"""
Merge h5 files given by glob into new file
This script assumes that datasets with a single dimension are meant to be names, and will NOT concatenate them into the final result

Marcel Rød
October 2021
"""
from __future__ import annotations

import gc
import glob
from pathlib import Path
from typing import List, Union, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
from h5py import File, Dataset, Group
from itertools import chain
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def flatten(l):
    return [item for sublist in l for item in sublist]


@dataclass
class Args:
    input_paths: List[Path]
    output_path: Path
    override_dtype: Optional[np.dtype]


@dataclass
class DatasetInfo:
    name: str
    should_merge: bool
    position: Position


@dataclass
class Position:
    """Traversal instructions to get to a dataset in the h5 file structure"""
    keys: List[str]

    def get_from_h5(self, h5_file: File):
        current_location: Union[File, Group, Dataset] = h5_file
        for k in self.keys:
            current_location = current_location[k]
        return current_location

    def get_merged_dataset_shape(self, h5_paths: List[Path]) -> Tuple[int, ...]:
        files = (File(p, 'r') for p in h5_paths)
        datasets = (self.get_from_h5(file) for file in files)
        shapes = list(dataset.shape for dataset in datasets)

        first_shape = shapes[0]
        return sum((shape[0] for shape in shapes), start=0), *first_shape[1:]

    def write_to_h5(self, h5_file: File, data: np.ndarray, compression: str = 'gzip'):
        assert len(self.keys) > 0
        current_location = h5_file

        # All but the last key are groups
        groups = self.keys[:-1]
        dataset_name = self.keys[-1]
        for group_name in groups:
            current_location = maybe_new_group(current_location, group_name)

        # Dataset should not exist yet
        current_location.create_dataset(name=dataset_name, data=data, compression=compression)
        print(f'Wrote array of shape {data.shape} to {self}')

    def write_to_h5_in_parts(self, h5_file: File, total_shape: Tuple[int, ...], data_generator: Generator[np.ndarray], compression: str = 'gzip', override_dtype: Optional[np.dtype] = None):
        assert len(self.keys) > 0
        current_location = h5_file

        # All but the last key are groups
        groups = self.keys[:-1]
        dataset_name = self.keys[-1]
        for group_name in groups:
            current_location = maybe_new_group(current_location, group_name)

        # Dataset should not exist yet
        new_dataset = current_location.create_dataset(name=dataset_name, shape=total_shape, compression=compression, dtype=override_dtype)

        start_row = 0
        for data in data_generator:
            num_rows = data.shape[0]
            new_dataset[start_row:start_row + num_rows] = data
            start_row += num_rows

    def __str__(self):
        return '.'.join(self.keys)


def maybe_new_group(h5_obj: Union[File, Group], group_name: str):
    if group_name in h5_obj.keys():
        return h5_obj[group_name]
    else:
        h5_obj.create_group(group_name)
        return h5_obj[group_name]


@dataclass
class Structure:
    datasets: List[DatasetInfo]


def _discover_recursive(h5_obj: Union[Dataset, Group, File], context: Optional[List[str]] = None) -> Union[List[DatasetInfo], DatasetInfo]:
    """Return a flat list with position and information about each dataset"""

    if context is None:
        context = []

    if isinstance(h5_obj, Dataset):
        should_merge = len(h5_obj.shape) > 1
        return DatasetInfo(name=h5_obj.name, should_merge=should_merge, position=Position(context))

    if isinstance(h5_obj, (Group, File)):
        return list(chain([_discover_recursive(h5_obj[k], context + [k]) for k in h5_obj.keys()]))

    raise ValueError(f'Type {type(h5_obj)} not recognized')


def discover_structure(h5_file: File) -> Structure:
    dataset_information = _discover_recursive(h5_file)
    return Structure(datasets=dataset_information)


def parse_args() -> Args:
    parser = ArgumentParser(description='Merge the h5 files at the first N paths into one file at the final path')
    parser.add_argument('input_paths', nargs='+', type=str, help='paths or globs for input files')
    parser.add_argument('output_path', type=str, help='the path of the output file')
    parser.add_argument('--override-dtype', type=np.dtype, default=None)
    args = parser.parse_args()

    input_paths = args.input_paths
    output_path = args.output_path
    override_dtype = args.override_dtype

    input_paths = flatten([glob.glob(input_p) for input_p in input_paths])
    input_paths = [Path(p).resolve() for p in input_paths]
    output_path = Path(output_path).resolve()

    arg_obj = Args(input_paths=input_paths,
                   output_path=output_path,
                   override_dtype=override_dtype)

    assert len(arg_obj.input_paths) > 0
    assert arg_obj.output_path is not None
    print(f'Merging files {arg_obj.input_paths} into the new file {arg_obj.output_path}')

    return arg_obj


def merge_all_datasets(args: Args, structure: Structure):
    input_paths = args.input_paths
    output_path = args.output_path

    with File(output_path, 'w') as output_file:
        for dataset_info in tqdm(structure.datasets, desc=f'Merging all datasets'):
            dataset_info: DatasetInfo
            if not dataset_info.should_merge:
                # When not merging, take only from the first dataset
                array = dataset_info.position.get_from_h5(File(input_paths[0], 'r'))[()]
                dataset_info.position.write_to_h5(output_file, data=array)
                del array
            else:
                # Write to the dataset in parts given by this generator
                def data_generator():
                    for ipath in tqdm(input_paths, desc=f'Fetching arrays for {dataset_info.name}'):
                        array = dataset_info.position.get_from_h5(File(ipath, 'r'))[()]
                        yield array
                dataset_info.position.write_to_h5_in_parts(h5_file=output_file,
                                                           total_shape=dataset_info.position.get_merged_dataset_shape(input_paths),
                                                           data_generator=data_generator(),
                                                           override_dtype=args.override_dtype)

            # Always run a gc sweep before the next run to keep memory usage in check
            gc.collect()


def merge_files(args: Args):
    with File(args.input_paths[0], 'r') as first_file:
        # Discover the structure of the input datasets
        structure = discover_structure(first_file)

    merge_all_datasets(args, structure)


def main():
    args = parse_args()
    merge_files(args)


if __name__ == '__main__':
    main()



