import argparse
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import AutoProcessor

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(func):
        return func

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelForImageTextToText

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_PROMPT = (
    "Task: Visual Place Recognition\n\n"
    "Analyze this image to identify regions most useful for recognizing this specific location "
    "across different times, viewpoints, and weather conditions.\n\n"
    "Focus on:\n"
    "- Permanent, stable structures (buildings, bridges, road layout)\n"
    "- Distinctive landmarks (unique facades, signs, monuments, poles)\n"
    "- Spatial layout and geometric relationships between structures\n"
    "- Reliable visual cues that persist over time\n\n"
    "Ignore:\n"
    "- Temporary objects (vehicles, pedestrians, construction equipment)\n"
    "- Appearance variations (lighting, shadows, weather, seasonal changes)\n\n"
    "Describe the key visual features that make this location uniquely identifiable."
)


CITY_PROMPT = (
    DEFAULT_PROMPT
    + "\n\n"
    + "For urban and city scenes, pay special attention to:\n"
    + "- Street topology and intersection structure (lane splits, medians, turn patterns, crosswalk layout)\n"
    + "- Persistent man-made elements (building facades, storefront arrangements, balconies, windows, awnings)\n"
    + "- Urban street furniture and infrastructure (traffic lights, poles, railings, barriers, bus stops, lamp posts)\n"
    + "- Distinctive signage and fixed roadside markers that are stable over time\n"
    + "- Spatial relationships among roads, sidewalks, curbs, bridges, and surrounding buildings\n\n"
    + "In dense urban scenes, prioritize stable architectural and road-layout cues over movable objects and short-term visual clutter."
)


HAWKINS_PROMPT = (
    DEFAULT_PROMPT
    + "\n\n"
    + "For indoor long-corridor scenes, pay special attention to:\n"
    + "- Stable corridor topology and geometry, including hallway direction, turns, intersections, side openings, and room entrances\n"
    + "- Persistent structural elements such as door frames, double doors, wall openings, windows, tiled walls, baseboards, ceiling panels, light fixtures, vents, and damaged ceiling sections\n"
    + "- Fixed local landmarks such as EXIT signs, room signs, warning signs, graffiti, wall marks, and other stationary man-made details\n"
    + "- Spatial relationships among doors, windows, corridor ends, wall segments, and ceiling structures\n"
    + "- Distinctive layout cues that help disambiguate one corridor segment from another in a highly repetitive indoor environment\n"
    + "- Regions that remain reliable despite fisheye distortion, robot-body occlusion, and strong exposure changes\n\n"
    + "For this dataset, prioritize structural and topological cues over overall appearance, and de-emphasize robot body parts, fisheye border artifacts, lighting hotspots, bright windows, underexposed regions, floor clutter, and other unstable local appearance changes."
)


SEVENTEENPLACES_PROMPT = (
    DEFAULT_PROMPT
    + "\n\n"
    + "For diverse indoor place-recognition scenes, pay special attention to:\n"
    + "- Room-level spatial layout and geometry, including corridor direction, room depth, doorway placement, wall openings, and stair structures\n"
    + "- Persistent indoor structures such as doors, windows, walls, ceiling panels, lighting patterns, vents, columns, railings, and built-in architectural elements\n"
    + "- Large fixed furnishings and functional fixtures such as shelves, cabinets, projector screens, podiums, counters, built-in desks, and stable furniture arrangements\n"
    + "- Distinctive floor and wall appearance patterns, including carpet designs, wall textures, wallpaper, molding, and other persistent decorative cues\n"
    + "- Spatial relationships among furniture groups, doors, shelves, screens, and other structural elements that distinguish one room or indoor area from another\n"
    + "- Stable scene cues that remain useful despite moderate viewpoint shifts, illumination differences, blur, and low-resolution image noise\n\n"
    + "For this dataset, prioritize room layout, architectural structure, and large stable indoor fixtures over small movable objects, monitor content, loose clutter, minor chair rearrangements, and local lighting or exposure changes."
)


NORDLAND_PROMPT = (
    DEFAULT_PROMPT
    + "\n\n"
    + "For the Nordland railway sequence, match this frame to the same physical position across summer and winter. "
    + "Pay special attention to:\n"
    + "- Railway geometry, including track curvature, switches, parallel tracks, crossings, and trackside alignment\n"
    + "- Persistent railway infrastructure such as poles, signals, signs, tunnels, bridges, platforms, barriers, and utility structures\n"
    + "- Stable terrain geometry, including mountain and hill silhouettes, rock cuttings, embankments, shorelines, and valley shape\n"
    + "- Long-range spatial relationships among the tracks, infrastructure, terrain, and permanent buildings\n\n"
    + "Strongly de-emphasize snow cover, vegetation color and foliage density, sky and illumination, seasonal texture, motion blur, "
    + "window reflections, and other appearance changes that do not identify the train's longitudinal position."
)


PROMPT_PRESETS = {
    "default": DEFAULT_PROMPT,
    "city": CITY_PROMPT,
    "hawkins": HAWKINS_PROMPT,
    "17places": SEVENTEENPLACES_PROMPT,
    "nordland": NORDLAND_PROMPT,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Optional text file containing one image path per line, relative to image_dir or absolute.",
    )
    parser.add_argument(
        "--msls-official-eval",
        action="store_true",
        help="Extract only images used by the official MSLS Challenge evaluator.",
    )
    parser.add_argument(
        "--mapillary-sls-root",
        default="data/benchmarks/mapillary_sls",
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_preset", type=str, default="default", choices=sorted(PROMPT_PRESETS.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--glob", type=str, default="*.jpg")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--exclude_subdirs", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_prompt(args):
    if args.prompt is not None:
        return args.prompt
    return PROMPT_PRESETS[args.prompt_preset]


def build_chat(processor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def list_images(image_dir: Path, pattern: str, recursive: bool, exclude_subdirs=None, image_list=None):
    if image_list is not None:
        resolved_root = image_dir.resolve()
        paths = []
        for raw_line in Path(image_list).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = image_dir / path
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(f"image_list path is outside image_dir: {path}") from exc
            if not resolved_path.is_file():
                raise FileNotFoundError(f"image_list entry does not exist: {path}")
            paths.append(resolved_path)
        return sorted(set(paths), key=lambda path: str(path.relative_to(resolved_root)))

    excluded = set(exclude_subdirs or [])
    iterator = image_dir.rglob(pattern) if recursive else image_dir.glob(pattern)
    paths = []
    resolved_root = image_dir.resolve()
    for path in iterator:
        if not path.is_file():
            continue
        if excluded:
            relative_parts = path.resolve().relative_to(resolved_root).parts
            if relative_parts and relative_parts[0] in excluded:
                continue
        paths.append(path)
    paths.sort()
    return paths


def list_msls_official_eval_images(image_dir: Path, mapillary_sls_root: Path):
    mapillary_sls_root = mapillary_sls_root.resolve()
    if str(mapillary_sls_root) not in sys.path:
        sys.path.insert(0, str(mapillary_sls_root))
    from mapillary_sls.datasets.msls import MSLS

    dataset = MSLS(
        str(image_dir.parent),
        cities="test",
        mode="val",
        task="im2im",
        subtask="all",
        seq_length=1,
        posDistThr=25,
    )
    paths = list(dataset.dbImages) + list(dataset.qImages[dataset.qIdx])
    resolved_root = image_dir.resolve()
    resolved_paths = []
    for path in paths:
        resolved_path = Path(path).resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"MSLS image is outside image_dir: {path}") from exc
        resolved_paths.append(resolved_path)
    return sorted(set(resolved_paths), key=lambda path: str(path.relative_to(resolved_root)))


def batched(sequence, batch_size: int):
    for start in range(0, len(sequence), batch_size):
        yield sequence[start:start + batch_size]


def load_image(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"failed to load image {image_path}: {exc}") from exc


def output_path_for_image(image_path: Path, image_root: Path, output_root: Path):
    relative = image_path.resolve().relative_to(image_root.resolve())
    return (output_root / relative).with_suffix(".pt")


def run_model_batch(model, processor, device: str, prompt_text: str, images):
    texts = [prompt_text] * len(images)
    with torch.inference_mode():
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1].detach().to("cpu")
        input_ids = inputs["input_ids"].detach().cpu()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.detach().cpu()
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.detach().cpu()
    return last_hidden, input_ids, attention_mask, image_grid_thw


def save_payload(out_path: Path, image_path: Path, prompt: str, hidden_states: torch.Tensor, input_ids: torch.Tensor, save_dtype, attention_mask=None, image_grid_thw=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "prompt": prompt,
        "hidden_states": hidden_states.to(dtype=save_dtype),
        "input_ids": input_ids,
    }
    if attention_mask is not None:
        payload["attention_mask"] = attention_mask
    if image_grid_thw is not None:
        payload["image_grid_thw"] = image_grid_thw
    torch.save(payload, out_path)


def resolve_worker_config(args):
    env_rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    rank = args.rank if args.rank is not None else env_rank
    world_size = args.world_size if args.world_size is not None else env_world_size
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    if not torch.cuda.is_available():
        return rank, world_size, None, "cpu"

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("CUDA is available but no visible GPUs were found")

    if args.gpu_id is not None:
        gpu_id = args.gpu_id
    elif "LOCAL_RANK" in os.environ:
        gpu_id = env_local_rank
    elif visible_gpu_count == 1:
        gpu_id = 0
    else:
        gpu_id = rank % visible_gpu_count

    if gpu_id < 0 or gpu_id >= visible_gpu_count:
        raise ValueError(f"gpu_id must be in [0, {visible_gpu_count}), got {gpu_id}")

    return rank, world_size, gpu_id, f"cuda:{gpu_id}"


@record
def main():
    args = parse_args()
    rank, world_size, gpu_id, device = resolve_worker_config(args)

    image_dir = Path(args.image_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    if args.msls_official_eval:
        if args.image_list is not None:
            raise ValueError("--msls-official-eval and --image_list are mutually exclusive")
        image_paths = list_msls_official_eval_images(
            image_dir, Path(args.mapillary_sls_root)
        )
    else:
        image_paths = list_images(
            image_dir,
            args.glob,
            args.recursive,
            args.exclude_subdirs,
            image_list=args.image_list,
        )
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise RuntimeError(f"no images found under {image_dir} with pattern {args.glob}")

    global_total = len(image_paths)
    image_paths = image_paths[rank::world_size]
    if not image_paths:
        print(f"[rank {rank}] no images assigned out of total={global_total}")
        return

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model_kwargs = {
        "dtype": resolve_dtype(args.dtype),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if gpu_id is not None:
        model_kwargs["device_map"] = {"": device}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    resolved_prompt = resolve_prompt(args)
    prompt_text = build_chat(processor, resolved_prompt)
    save_dtype = resolve_dtype(args.save_dtype)

    total = len(image_paths)
    success = 0
    skipped = 0
    failed = 0

    print(f"[rank {rank}] assigned={total} global_total={global_total} device={device}")

    progress_bar = tqdm(
        total=total,
        desc=f"Rank {rank}",
        dynamic_ncols=True,
        position=rank,
    )
    try:
        for batch_index, batch_paths in enumerate(batched(image_paths, args.batch_size), start=1):
            batch_start_time = perf_counter()
            batch_skipped = 0
            batch_failed = 0
            uncached_paths = []
            output_paths = []
            images = []

            for image_path in batch_paths:
                out_path = output_path_for_image(image_path, image_dir, output_root)
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    batch_skipped += 1
                    continue
                try:
                    image = load_image(image_path)
                except RuntimeError as exc:
                    failed += 1
                    batch_failed += 1
                    tqdm.write(f"[warn] {exc}")
                    continue
                uncached_paths.append(image_path)
                output_paths.append(out_path)
                images.append(image)

            if images:
                try:
                    last_hidden, input_ids, attention_mask, image_grid_thw = run_model_batch(
                        model=model,
                        processor=processor,
                        device=device,
                        prompt_text=prompt_text,
                        images=images,
                    )

                    for sample_idx, (image_path, out_path) in enumerate(zip(uncached_paths, output_paths)):
                        sample_attention_mask = None if attention_mask is None else attention_mask[sample_idx]
                        sample_image_grid_thw = None if image_grid_thw is None else image_grid_thw[sample_idx]
                        save_payload(
                            out_path=out_path,
                            image_path=image_path,
                            prompt=resolved_prompt,
                            hidden_states=last_hidden[sample_idx],
                            input_ids=input_ids[sample_idx],
                            save_dtype=save_dtype,
                            attention_mask=sample_attention_mask,
                            image_grid_thw=sample_image_grid_thw,
                        )
                        success += 1
                except Exception as exc:
                    tqdm.write(
                        f"[rank {rank} batch {batch_index}] batch_retry_due_to_error={type(exc).__name__}: {exc}"
                    )
                    tqdm.write(traceback.format_exc())
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

                    for image_path, out_path, image in zip(uncached_paths, output_paths, images):
                        try:
                            last_hidden, input_ids, attention_mask, image_grid_thw = run_model_batch(
                                model=model,
                                processor=processor,
                                device=device,
                                prompt_text=prompt_text,
                                images=[image],
                            )
                            save_payload(
                                out_path=out_path,
                                image_path=image_path,
                                prompt=resolved_prompt,
                                hidden_states=last_hidden[0],
                                input_ids=input_ids[0],
                                save_dtype=save_dtype,
                                attention_mask=None if attention_mask is None else attention_mask[0],
                                image_grid_thw=None if image_grid_thw is None else image_grid_thw[0],
                            )
                            success += 1
                        except Exception as single_exc:
                            failed += 1
                            batch_failed += 1
                            tqdm.write(
                                f"[rank {rank}] failed_image={image_path} error={type(single_exc).__name__}: {single_exc}"
                            )
                            tqdm.write(traceback.format_exc())
                            if device.startswith("cuda"):
                                torch.cuda.empty_cache()

            batch_elapsed = perf_counter() - batch_start_time
            progress_bar.update(len(batch_paths))
            progress_bar.set_postfix(
                processed=success,
                skipped=skipped,
                failed=failed,
                batch_s=f"{batch_elapsed:.2f}",
            )
            tqdm.write(
                f"[rank {rank} batch {batch_index}] size={len(batch_paths)} processed={len(uncached_paths)} "
                f"skipped_in_batch={batch_skipped} failed_in_batch={batch_failed} elapsed={batch_elapsed:.2f}s"
            )
    finally:
        progress_bar.close()

    print(
        f"[rank {rank} done] processed={success} skipped={skipped} failed={failed} total={total} output_root={output_root}"
    )


if __name__ == "__main__":
    main()
