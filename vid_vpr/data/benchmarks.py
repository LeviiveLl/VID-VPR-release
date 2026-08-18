
import csv
import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
from pathlib import Path
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SUPPORTED_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

# Dataset protocols define dataset-specific localization radii for unstructured
# pose-based benchmarks. These must not inherit the 25 m urban VPR default.
OFFICIAL_POSITIVE_DISTANCE_METERS = {
    "hawkins_long_corridor": 8.0,
    "laurel_caverns": 8.0,
}


def resolve_positive_distance(dataset_name, fallback_distance):
    return OFFICIAL_POSITIVE_DISTANCE_METERS.get(dataset_name, float(fallback_distance))


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def _image_sort_key(path):
    stem = Path(path).stem
    try:
        return (0, int(stem), str(path))
    except ValueError:
        return (1, str(path))


def _gather_image_paths(folder):
    paths = []
    for pattern in SUPPORTED_IMAGE_EXTENSIONS:
        paths.extend(glob(join(folder, "**", pattern), recursive=True))
    return sorted(set(paths), key=_image_sort_key)


def _image_path_from_name(folder, name):
    path = Path(folder) / name
    if path.exists():
        return str(path)
    name_path = Path(name)
    for ext in [".jpg", ".jpeg", ".png"]:
        path = Path(folder) / name_path.with_suffix(ext)
        if path.exists():
            return str(path)
    stem = name_path.stem
    for ext in [".jpg", ".jpeg", ".png"]:
        path = Path(folder) / f"{stem}{ext}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Image {name} not found in {folder}")


def _load_index_ground_truth(gt_path, queries_count, database_count, query_ids=None):
    gt_data = np.load(gt_path, allow_pickle=True)
    if len(gt_data) >= queries_count and queries_count > 0:
        tail = gt_data[-queries_count:]
        if all(hasattr(item, "__len__") and len(item) == 2 for item in tail):
            gt_rows = tail
        else:
            gt_rows = gt_data[:queries_count]
    else:
        gt_rows = gt_data

    if query_ids is not None:
        row_by_query_id = {}
        for item in gt_data:
            if hasattr(item, "__len__") and len(item) == 2:
                row_by_query_id[int(item[0])] = item[1]
        positives_per_query = []
        for query_id in query_ids:
            if int(query_id) not in row_by_query_id:
                raise ValueError(f"Missing ground-truth entry for query id {query_id} in {gt_path}")
            positives = row_by_query_id[int(query_id)]
            positives = np.asarray(list(positives), dtype=int)
            positives = positives[(positives >= 0) & (positives < database_count)]
            if positives.size == 0:
                raise ValueError(f"No valid positives for query id {query_id} in {gt_path}")
            positives_per_query.append(positives)
        return positives_per_query

    if len(gt_rows) != queries_count:
        raise ValueError(f"Ground-truth length mismatch in {gt_path}: expected {queries_count}, got {len(gt_rows)}")

    positives_per_query = []
    for row_index, item in enumerate(gt_rows):
        if hasattr(item, "__len__") and len(item) == 2 and not isinstance(item, (str, bytes)):
            positives = item[1]
        else:
            positives = [item]
        positives = np.asarray(list(positives), dtype=int)
        positives = positives[(positives >= 0) & (positives < database_count)]
        if positives.size == 0:
            raise ValueError(f"No valid positives for query {row_index} in {gt_path}")
        positives_per_query.append(positives)
    return positives_per_query


def _read_xy_csv(csv_path, image_folder):
    paths = []
    positions = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            name = row["name"]
            paths.append(_image_path_from_name(image_folder, name))
            positions.append([float(row["easting"]), float(row["northing"])])
    return paths, np.asarray(positions, dtype=float)


def _read_topk_ground_truth(
    csv_path, queries_count, database_count, max_positive_rank=None
):
    positives_per_query = [None] * queries_count
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        topk_columns = sorted(
            [name for name in reader.fieldnames if name.startswith("top_") and name.endswith("_ref_ind")],
            key=lambda name: int(name.split("_")[1]),
        )
        if max_positive_rank is not None:
            topk_columns = topk_columns[:max_positive_rank]
        for row in reader:
            query_index = int(row["query_ind"])
            positives = []
            for column in topk_columns:
                value = row.get(column, "")
                if value != "":
                    positives.append(int(value))
            positives = np.asarray(positives, dtype=int)
            positives = positives[(positives >= 0) & (positives < database_count)]
            if positives.size == 0:
                raise ValueError(f"No valid positives for query {query_index} in {csv_path}")
            positives_per_query[query_index] = positives
    if any(item is None for item in positives_per_query):
        raise ValueError(f"Missing ground-truth rows in {csv_path}")
    return positives_per_query


def _read_baidu_pose(camera_path):
    lines = Path(camera_path).read_text().strip().splitlines()
    return np.asarray([float(value) for value in lines[7].split()[:2]], dtype=float)


def _read_baidu_split(image_folder, camera_folder):
    paths = _gather_image_paths(image_folder)
    positions = []
    for path in paths:
        camera_path = Path(camera_folder) / f"{Path(path).stem}.camera"
        if not camera_path.exists():
            raise FileNotFoundError(f"Missing Baidu camera pose for {path}: {camera_path}")
        positions.append(_read_baidu_pose(camera_path))
    return paths, np.asarray(positions, dtype=float)


def _read_msls_split(split_folder):
    csv_path = Path(split_folder) / "postprocessed.csv"
    image_folder = Path(split_folder) / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"MSLS metadata not found: {csv_path}")
    paths = []
    positions = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            key = row["key"]
            paths.append(_image_path_from_name(image_folder, key))
            positions.append([float(row["easting"]), float(row["northing"])])
    return paths, np.asarray(positions, dtype=float)


def _read_oxford_split(root_folder, split_name):
    image_folder = Path(root_folder) / "oxDataPart" / split_name
    pose_path = Path(root_folder) / "oxDataPart" / f"{split_name}.txt"
    if not image_folder.exists() or not pose_path.exists():
        raise FileNotFoundError(f"Oxford RobotCar split not found: {split_name}")
    paths = _gather_image_paths(image_folder)
    pose_rows = np.loadtxt(pose_path, delimiter=",", comments="#")
    if len(paths) != len(pose_rows):
        raise ValueError(f"Oxford RobotCar split length mismatch for {split_name}: images={len(paths)}, poses={len(pose_rows)}")
    return paths, np.asarray(pose_rows[:, :2], dtype=float)


def _read_nordland_split(root_folder, season, image_names):
    candidates = [
        Path(root_folder) / "data" / season,
        Path(root_folder) / season,
    ]
    image_folder = next((path for path in candidates if path.exists()), None)
    if image_folder is None:
        raise FileNotFoundError(f"Nordland season images not found for {season}; expected one of {candidates}")
    paths = [_image_path_from_name(image_folder, name) for name in image_names]
    return paths


def _nordland_available_names(root_folder, season, image_names):
    candidates = [
        Path(root_folder) / "data" / season,
        Path(root_folder) / season,
    ]
    image_folder = next((path for path in candidates if path.exists()), None)
    if image_folder is None:
        raise FileNotFoundError(f"Nordland season images not found for {season}; expected one of {candidates}")
    available = {Path(path).name for path in _gather_image_paths(image_folder)}
    return [name for name in image_names if name in available]


def _nordland_sampled_query_indices(database_count):
    return list(range(0, database_count, 10))


def _nordland_window_positives(query_indices, database_count):
    positives_per_query = []
    for index in query_indices:
        start = max(0, index - 1)
        stop = min(database_count, index + 2)
        positives_per_query.append(np.arange(start, stop, dtype=int))
    return positives_per_query


def _read_tokyo247(root_folder):
    import scipy.io as sio

    mat = sio.loadmat(Path(root_folder) / "tokyo247.mat", squeeze_me=True, struct_as_record=False)
    db_struct = mat["dbStruct"]
    database_paths = [_image_path_from_name(root_folder, name) for name in np.asarray(db_struct.dbImageFns).tolist()]
    queries_paths = [_image_path_from_name(Path(root_folder) / "247query_subset_v2", name) for name in np.asarray(db_struct.qImageFns).tolist()]
    database_utms = np.asarray(db_struct.utmDb, dtype=float).T
    queries_utms = np.asarray(db_struct.utmQ, dtype=float).T
    if len(database_paths) != int(db_struct.numImages):
        raise ValueError(f"Tokyo247 database count mismatch: expected {db_struct.numImages}, got {len(database_paths)}")
    if len(queries_paths) != int(db_struct.numQueries):
        raise ValueError(f"Tokyo247 query count mismatch: expected {db_struct.numQueries}, got {len(queries_paths)}")
    return database_paths, database_utms, queries_paths, queries_utms


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images, 
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images                  = torch.cat([e[0] for e in batch])
    triplets_local_indexes  = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        needs_vlm_cache = getattr(args, "use_vlm_crossattn", False)
        self.vlm_cache_dir = None
        if needs_vlm_cache:
            self.vlm_cache_dir = getattr(args, 'eval_vlm_cache_dir', None)
            if self.vlm_cache_dir is None:
                self.vlm_cache_dir = getattr(args, 'vlm_cache_dir', None)

        self.resize = args.resize
        self.test_method = args.test_method

        standard_dataset_folder = join(datasets_folder, dataset_name, "images", split)
        standard_database_folder = join(standard_dataset_folder, "database")
        standard_queries_folder = join(standard_dataset_folder, "queries")
        raw_dataset_folder = join(datasets_folder, dataset_name)
        raw_database_folder = join(raw_dataset_folder, "db_images")
        raw_queries_folder = join(raw_dataset_folder, "q_images")
        raw_pose_path = join(raw_dataset_folder, "pose_topic_list.npy")
        raw_ref_folder = join(raw_dataset_folder, "ref")
        raw_query_folder = join(raw_dataset_folder, "query")
        corrected_gt_path = join(raw_dataset_folder, "my_ground_truth_new.npy")
        original_gt_path = join(raw_dataset_folder, "ground_truth_new.npy")

        if dataset_name.startswith("Nordland:") or dataset_name.startswith("NordlandFull:"):
            prefix, database_season, query_season = dataset_name.split(":", 2)
            root_folder = Path(datasets_folder) / "Nordland"
            names_path = root_folder / "dataset_imageNames" / "nordland_imageNames.txt"
            image_names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
            database_names = set(_nordland_available_names(root_folder, database_season, image_names))
            query_names = set(_nordland_available_names(root_folder, query_season, image_names))
            image_names = [name for name in image_names if name in database_names and name in query_names]
            if len(image_names) == 0:
                raise FileNotFoundError(f"No overlapping Nordland images found for {database_season}->{query_season}")
            self.dataset_folder = str(root_folder)
            self.database_paths = _read_nordland_split(root_folder, database_season, image_names)
            if prefix == "NordlandFull":
                query_indices = list(range(len(image_names)))
            else:
                query_indices = _nordland_sampled_query_indices(len(image_names))
            query_image_names = [image_names[index] for index in query_indices]
            self.queries_paths = _read_nordland_split(root_folder, query_season, query_image_names)
            self.database_utms = np.zeros((len(self.database_paths), 2), dtype=float)
            self.queries_utms = np.zeros((len(self.queries_paths), 2), dtype=float)
            if prefix == "NordlandFull":
                self.soft_positives_per_query = [np.asarray([index], dtype=int) for index in query_indices]
            else:
                self.soft_positives_per_query = _nordland_window_positives(query_indices, len(self.database_paths))
        elif dataset_name.startswith("gardens:") or dataset_name == "gardens":
            parts = dataset_name.split(":")
            database_split = parts[1] if len(parts) > 1 else "day_right"
            query_split = parts[2] if len(parts) > 2 else "night_right"
            root_folder = Path(datasets_folder) / "gardens"
            self.dataset_folder = str(root_folder)
            self.database_paths = _gather_image_paths(root_folder / database_split)
            self.queries_paths = _gather_image_paths(root_folder / query_split)
            self.database_utms = np.zeros((len(self.database_paths), 2), dtype=float)
            self.queries_utms = np.zeros((len(self.queries_paths), 2), dtype=float)
            self.soft_positives_per_query = _load_index_ground_truth(root_folder / "gardens_gt.npy", len(self.queries_paths), len(self.database_paths))
        elif dataset_name.startswith("Oxford_Robotcar:") or dataset_name == "Oxford_Robotcar":
            parts = dataset_name.split(":")
            database_split = parts[1] if len(parts) > 1 else "1-s"
            query_split = parts[2] if len(parts) > 2 else "2-s"
            root_folder = Path(datasets_folder) / "Oxford_Robotcar"
            self.dataset_folder = str(root_folder)
            self.database_paths, self.database_utms = _read_oxford_split(root_folder, database_split)
            self.queries_paths, self.queries_utms = _read_oxford_split(root_folder, query_split)
        elif dataset_name == "baidu_datasets":
            root_folder = Path(datasets_folder) / "baidu_datasets"
            self.dataset_folder = str(root_folder)
            self.database_paths, self.database_utms = _read_baidu_split(root_folder / "training_images_undistort", root_folder / "training_gt")
            self.queries_paths, self.queries_utms = _read_baidu_split(root_folder / "query_images_undistort", root_folder / "query_gt")
        elif dataset_name == "test_40_midref_rot0":
            root_folder = Path(datasets_folder) / dataset_name
            self.dataset_folder = str(root_folder)
            self.database_paths, self.database_utms = _read_xy_csv(root_folder / "reference.csv", root_folder / "reference_images")
            self.queries_paths, self.queries_utms = _read_xy_csv(root_folder / "query.csv", root_folder / "query_images")
            self.ground_truth_top_k = 5
            self.soft_positives_per_query = _read_topk_ground_truth(
                root_folder / "gt_matches.csv",
                len(self.queries_paths),
                len(self.database_paths),
                max_positive_rank=self.ground_truth_top_k,
            )
        elif dataset_name == "VPAir":
            root_folder = Path(datasets_folder) / dataset_name
            self.dataset_folder = str(root_folder)
            self.database_paths = _gather_image_paths(root_folder / "reference_views")
            self.queries_paths = _gather_image_paths(root_folder / "queries")
            query_ids = [int(Path(path).stem) for path in self.queries_paths]
            self.database_utms = np.zeros((len(self.database_paths), 2), dtype=float)
            self.queries_utms = np.zeros((len(self.queries_paths), 2), dtype=float)
            self.soft_positives_per_query = _load_index_ground_truth(root_folder / "vpair_gt.npy", len(self.queries_paths), len(self.database_paths), query_ids=query_ids)
        elif dataset_name == "eiffel":
            root_folder = Path(datasets_folder) / dataset_name
            self.dataset_folder = str(root_folder)
            self.database_paths = _gather_image_paths(root_folder / "db_images")
            self.queries_paths = _gather_image_paths(root_folder / "q_images")
            self.database_utms = np.zeros((len(self.database_paths), 2), dtype=float)
            self.queries_utms = np.zeros((len(self.queries_paths), 2), dtype=float)
            self.soft_positives_per_query = _load_index_ground_truth(root_folder / "eiffel_gt.npy", len(self.queries_paths), len(self.database_paths))
        elif dataset_name == "tokyo247":
            root_folder = Path(datasets_folder) / "tokyo247"
            self.dataset_folder = str(root_folder)
            self.database_paths, self.database_utms, self.queries_paths, self.queries_utms = _read_tokyo247(root_folder)
        elif "msls/" in dataset_name and os.path.exists(join(raw_dataset_folder, "database", "images")) and os.path.exists(join(raw_dataset_folder, "query", "images")):
            self.dataset_folder = raw_dataset_folder
            self.database_paths, self.database_utms = _read_msls_split(Path(raw_dataset_folder) / "database")
            self.queries_paths, self.queries_utms = _read_msls_split(Path(raw_dataset_folder) / "query")
        elif os.path.exists(standard_database_folder) and os.path.exists(standard_queries_folder):
            self.dataset_folder = standard_dataset_folder
            self.database_paths = _gather_image_paths(standard_database_folder)
            self.queries_paths = _gather_image_paths(standard_queries_folder)
            self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
            self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        elif os.path.exists(raw_database_folder) and os.path.exists(raw_queries_folder) and os.path.exists(raw_pose_path):
            self.dataset_folder = raw_dataset_folder
            self.database_paths = _gather_image_paths(raw_database_folder)
            self.queries_paths = _gather_image_paths(raw_queries_folder)
            if len(self.database_paths) == 0:
                raise FileNotFoundError(f"No database images found in {raw_database_folder}")
            if len(self.queries_paths) == 0:
                raise FileNotFoundError(f"No query images found in {raw_queries_folder}")

            pose_data = np.load(raw_pose_path)
            xy_positions = np.asarray(pose_data[:, :2], dtype=float)
            database_indices = np.array([int(Path(path).stem) for path in self.database_paths], dtype=int)
            query_indices = np.array([int(Path(path).stem) for path in self.queries_paths], dtype=int)

            if database_indices.max(initial=-1) >= len(xy_positions):
                raise ValueError(f"Database image index exceeds pose array length in {raw_pose_path}")
            if query_indices.max(initial=-1) >= len(xy_positions):
                raise ValueError(f"Query image index exceeds pose array length in {raw_pose_path}")

            self.database_utms = xy_positions[database_indices]
            self.queries_utms = xy_positions[query_indices]
        elif os.path.exists(raw_ref_folder) and os.path.exists(raw_query_folder) and (os.path.exists(corrected_gt_path) or os.path.exists(original_gt_path)):
            self.dataset_folder = raw_dataset_folder
            self.database_paths = _gather_image_paths(raw_ref_folder)
            self.queries_paths = _gather_image_paths(raw_query_folder)
            if len(self.database_paths) == 0:
                raise FileNotFoundError(f"No database images found in {raw_ref_folder}")
            if len(self.queries_paths) == 0:
                raise FileNotFoundError(f"No query images found in {raw_query_folder}")

            gt_path = corrected_gt_path if os.path.exists(corrected_gt_path) else original_gt_path
            gt_data = np.load(gt_path, allow_pickle=True)
            if len(gt_data) != len(self.queries_paths):
                raise ValueError(f"Ground-truth length mismatch in {gt_path}: expected {len(self.queries_paths)}, got {len(gt_data)}")

            self.database_utms = np.zeros((len(self.database_paths), 2), dtype=float)
            self.queries_utms = np.zeros((len(self.queries_paths), 2), dtype=float)
            self.soft_positives_per_query = []
            for row_index, item in enumerate(gt_data):
                query_index, positive_indices = item
                if int(query_index) != row_index:
                    raise ValueError(f"Ground-truth query index mismatch in {gt_path} at row {row_index}: found {query_index}")

                positives = np.asarray(list(positive_indices), dtype=int)
                positives = positives[(positives >= 0) & (positives < len(self.database_paths))]
                if positives.size == 0:
                    raise ValueError(f"No valid positives for query {row_index} in {gt_path}")
                self.soft_positives_per_query.append(positives)
        else:
            raise FileNotFoundError(
                f"Could not find a supported dataset layout for {dataset_name}. Checked {standard_dataset_folder} and {raw_dataset_folder}"
            )

        if not hasattr(self, "soft_positives_per_query"):
            positive_distance = resolve_positive_distance(
                dataset_name, args.val_positive_dist_threshold
            )
            self.positive_distance_m = positive_distance
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                                radius=positive_distance,
                                                                 return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
    
    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        
        if self.vlm_cache_dir is not None:
            img_path = self.images_paths[index]
            cache_name = Path(img_path).with_suffix('.pt').name
            # Search in vlm_cache_dir (flat or with subdirectory)
            cache_path = Path(self.vlm_cache_dir) / cache_name
            if not cache_path.exists():
                # Try with parent directory name as subdirectory
                parent_name = Path(img_path).parent.name
                cache_path = Path(self.vlm_cache_dir) / parent_name / cache_name
            if not cache_path.exists():
                # Preserve the dataset-relative hierarchy for deeply nested
                # benchmarks such as Tokyo247.
                try:
                    relative_path = Path(img_path).resolve().relative_to(
                        Path(self.dataset_folder).resolve()
                    )
                    cache_path = (
                        Path(self.vlm_cache_dir) / relative_path
                    ).with_suffix(".pt")
                except ValueError:
                    pass
            if cache_path.exists():
                payload = torch.load(cache_path, map_location='cpu', weights_only=True)
                return img, index, payload['hidden_states'], payload['attention_mask']
            else:
                raise FileNotFoundError(f"VLM cache not found for: {img_path}")
        return img, index
    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = transforms.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img
    
    def __len__(self):
        return len(self.images_paths)
    def __repr__(self):
        return  (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >")
    def get_positives(self):
        return self.soft_positives_per_query


class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]
