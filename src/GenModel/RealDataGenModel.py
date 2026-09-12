import glob
import hashlib
import os
import queue
import random
import threading
from dataclasses import dataclass

import torch
from PIL import Image
from torch.distributions.normal import Normal
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from src.Abstract.AbsMetricsLogger import AbsMetricsLogger
from src.DataStructures import ActionData, ImageData

_IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
_ENCODER_NAME = "google/vit-base-patch16-224-in21k"
_CACHE_DIR = os.path.join(".pbrl_cache", "RealDataGenModel")
_SANITY_CHECK_SAMPLE_SIZE = 15
_PREFETCH_BATCHES = 2
_CACHE_FLUSH_INTERVAL_ENTRIES = 5000
_LOG_PREFIX = "[RealDataGenModel cache]"
_NEAREST_NEIGHBOR_CHUNK_SIZE = 20000


class RealDataGenModel(object):
    """Gen model backed by real images instead of a generator network.

    Encodes every image in `image_folder` with a ViT encoder (same model
    as CosDistFeedback), fits a diagonal Normal distribution to the
    resulting encodings, and generates by looking up the nearest
    precomputed encoding for each requested action.
    """

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        image_folder: str
        image_size: int
        batch_size: int
        device: str | None

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return RealDataGenModel(
            image_folder=conf.image_folder,
            image_size=conf.image_size,
            batch_size=conf.batch_size,
            device=conf.device,
        )

    def __init__(
        self,
        image_folder: str,
        image_size: int = 128,
        batch_size: int = 64,
        device: str | None = None,
    ):
        """
        Args:
            image_folder (str): path to a folder of real images
            image_size (int, optional): images are resized to a square of
                this side length before encoding. Defaults to 128.
            batch_size (int, optional): batch size used while precomputing
                encodings. Defaults to 64.
            device (str | None, optional): device to precompute encodings
                on and to serve generate()/sample_random_actions() from.
                Defaults to None (cpu).
        """

        self.image_folder = image_folder
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device

        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if device is not None:
            # transformers ships no py.typed stubs; basedpyright misresolves
            # PreTrainedModel.to()'s overloads against __call__.
            self.model = self.model.to(device)  # pyright: ignore[reportArgumentType]
        self.model = self.model.eval()

        self._image_paths = self._list_image_paths(image_folder)
        self._encodings = self._get_encodings(batch_size)

        mean = self._encodings.mean(dim=0)
        std = self._encodings.std(dim=0).clamp_min(1e-6)
        self.dist = Normal(mean, std, validate_args=None)

    def _list_image_paths(self, image_folder: str) -> list[str]:
        """Returns the sorted list of image paths in image_folder, used as
        part of the encoding cache key/sanity check below. Images
        themselves are loaded lazily in batches (see _load_image_batch)
        rather than all at once, since image_folder can hold hundreds of
        thousands of images.
        """

        paths = []
        for pattern in _IMAGE_GLOBS:
            paths.extend(glob.glob(os.path.join(image_folder, pattern)))
        paths.sort()

        if not paths:
            raise Exception(f"No images found in {image_folder}")

        return paths

    def _load_image_batch(self, paths: list[str], image_size: int) -> torch.Tensor:
        """Loads the given image paths into a single [N, C, H, W] tensor,
        resized to (image_size, image_size).
        """

        to_tensor = transforms.ToTensor()
        image_tensors: list[torch.Tensor] = [
            to_tensor(Image.open(p).convert("RGB").resize((image_size, image_size)))
            for p in paths
        ]

        return torch.stack(image_tensors)

    def _cache_file(self) -> str:
        """Returns the cache file path for the current image_folder/
        image_size/encoder combination.
        """

        key = f"{os.path.abspath(self.image_folder)}|{self.image_size}|{_ENCODER_NAME}"
        digest = hashlib.sha256(key.encode()).hexdigest()

        return os.path.join(_CACHE_DIR, f"{digest}.pt")

    def _get_encodings(self, batch_size: int) -> torch.Tensor:
        """Loads precomputed encodings from the on-disk cache if present
        and passing a sanity check against a handful of freshly re-encoded
        images from within it. Any current images the cache doesn't cover
        (new since it was written, or left over from an interrupted run -
        see _encode_images) are encoded fresh and merged in by path, so an
        interrupted run resumes instead of starting over. Falls back to a
        full (re-)encode if there's no cache or it fails its sanity check.
        """

        cache_file = self._cache_file()
        cached = self._load_cache(cache_file)

        if cached is None or not self._cache_passes_sanity_check(cached, batch_size):
            return self._encode_images(
                self._image_paths, batch_size, cache_file=cache_file
            )

        return self._encode_images(
            self._image_paths,
            batch_size,
            cache_file=cache_file,
            resume_cache=cached,
        )

    def _load_cache(self, cache_file: str) -> dict | None:
        """Returns the cached dict if it exists, its encoder/image_size
        metadata matches the current setup, and it's internally consistent
        (one encoding per listed path) - regardless of how many images it
        covers or what order they're in. `_encode_images` reconciles
        cached paths against the currently requested ones by identity
        (path string), not position, so a cache from a folder that's since
        had images added, removed, or reordered is still resumed from for
        whatever overlap remains; entries for images that no longer exist
        are simply never looked up. Returns None if the file is missing or
        the encoder/image_size don't match, or if `encodings`/`paths` are
        inconsistent with each other (e.g. a checkpoint write that didn't
        complete).
        """

        if not os.path.exists(cache_file):
            print(f"{_LOG_PREFIX} no cache file at {cache_file}")
            return None

        cached = torch.load(cache_file, weights_only=True)

        if cached["encoder_name"] != _ENCODER_NAME:
            print(
                f"{_LOG_PREFIX} ignoring {cache_file}: encoder mismatch "
                f"(cached={cached['encoder_name']!r}, current={_ENCODER_NAME!r})"
            )
            return None

        if cached["image_size"] != self.image_size:
            print(
                f"{_LOG_PREFIX} ignoring {cache_file}: image_size mismatch "
                f"(cached={cached['image_size']}, current={self.image_size})"
            )
            return None

        if cached["encodings"].shape[0] != len(cached["paths"]):
            print(
                f"{_LOG_PREFIX} ignoring {cache_file}: inconsistent cache "
                f"({cached['encodings'].shape[0]} encodings for "
                f"{len(cached['paths'])} paths)"
            )
            return None

        print(
            f"{_LOG_PREFIX} found cache at {cache_file} with "
            f"{len(cached['paths'])} encoded images"
        )
        return cached

    def _cache_passes_sanity_check(self, cached: dict, batch_size: int) -> bool:
        """Re-encodes a random sample of images from within the cache's own
        `paths` (whatever it covers - every image for a complete cache, or
        just the images encoded so far for a partial checkpoint) and
        compares against the cached encodings at the same indices, to
        catch stale caches that metadata alone wouldn't (e.g. image files
        edited in place).

        Compared by cosine similarity (same metric CosDistFeedback uses
        for ViT embeddings) rather than raw allclose: re-running the same
        images through a 12-layer transformer on CPU doesn't reproduce
        bit-identical activations run to run (thread-scheduling changes
        floating-point summation order), so an absolute tolerance on raw
        hidden states flags harmless numerical noise as staleness. Cosine
        similarity is scale-invariant, so it easily tolerates that noise
        while still catching an embedding that's actually for a different
        image (cosine similarity would be far from 1.0, not just slightly
        off).
        """

        cached_paths = cached["paths"]
        sample_size = min(_SANITY_CHECK_SAMPLE_SIZE, len(cached_paths))
        indices = random.sample(range(len(cached_paths)), sample_size)

        sample_paths = [cached_paths[i] for i in indices]
        fresh = self._encode_images(sample_paths, batch_size)
        cached_subset = cached["encodings"][indices].to(fresh.device)

        cos_sim = torch.nn.functional.cosine_similarity(fresh, cached_subset, dim=1)
        min_cos_sim = cos_sim.min().item()
        passed = min_cos_sim > 0.999

        if passed:
            print(
                f"{_LOG_PREFIX} sanity check on {sample_size} sampled images: "
                f"passed (min cosine similarity {min_cos_sim:.6f})"
            )
        else:
            max_abs_diff = (fresh - cached_subset).abs().max().item()
            print(
                f"{_LOG_PREFIX} sanity check on {sample_size} sampled images: "
                f"FAILED (min cosine similarity {min_cos_sim:.6f}, max abs diff "
                f"{max_abs_diff:.6f}) - treating cache as stale"
            )
        return passed

    def _save_cache(
        self, cache_file: str, encodings: torch.Tensor, paths: list[str]
    ) -> None:
        """Writes encodings plus the metadata needed to validate/sanity
        check them on a later run to the cache file. `paths` is normally
        the full `self._image_paths`, but `_encode_images` also uses this
        to write periodic partial checkpoints during a long encode run -
        `paths` need not be in any particular order or cover every image;
        `_load_cache`/`_encode_images` reconcile it against the current
        image list by identity on the next run (see there).
        """

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(
            {
                "encodings": encodings.detach().cpu(),
                "paths": paths,
                "image_size": self.image_size,
                "encoder_name": _ENCODER_NAME,
            },
            cache_file,
        )

    @torch.no_grad()
    def _encode_images(
        self,
        paths: list[str],
        batch_size: int,
        cache_file: str | None = None,
        resume_cache: dict | None = None,
    ) -> torch.Tensor:
        """Runs the ViT encoder over images in batches, taking the CLS
        token embedding (last_hidden_state[:, 0]) as each image's encoding
        - same extraction as CosDistFeedback. A background thread loads and
        resizes the next _PREFETCH_BATCHES batches from disk while the
        current batch runs through the model, so disk I/O and inference
        overlap instead of the model sitting idle during every read. The
        queue is bounded so the loader can't run arbitrarily far ahead and
        pile up unread batches in memory.

        Each batch's encoding is detached and moved to the CPU right away
        (never returned with a grad_fn), so it doesn't sit resident in GPU
        memory while later batches are still being encoded. If `cache_file`
        is given, progress is additionally checkpointed to disk roughly
        every _CACHE_FLUSH_INTERVAL_ENTRIES images, with `torch.cuda.
        empty_cache()` run right after so freed GPU memory is actually
        returned to the device rather than just held by torch's allocator.

        If `resume_cache` is given (a dict with `paths`/`encodings`, as
        validated by `_load_cache`), any path in `paths` that's already
        present in `resume_cache["paths"]` is reused instead of
        re-encoded - matched by the path string itself, not position, so
        the resumed cache can be in any order or cover any subset of
        `paths`. Only the remaining paths are actually encoded. The
        returned tensor's rows are always reordered to line up with
        `paths`, regardless of what order the resumed/new pieces came in.
        """

        resume_paths: list[str] = [] if resume_cache is None else resume_cache["paths"]
        resume_encodings = None if resume_cache is None else resume_cache["encodings"]
        already_encoded_paths = set(resume_paths)

        remaining_paths = [p for p in paths if p not in already_encoded_paths]

        if resume_cache is not None:
            print(
                f"{_LOG_PREFIX} resuming: {len(paths) - len(remaining_paths)} of "
                f"{len(paths)} images already cached, encoding {len(remaining_paths)} more"
            )
        elif cache_file is not None:
            print(f"{_LOG_PREFIX} no cache resumed, encoding all {len(paths)} images")

        batch_path_lists = [
            remaining_paths[start : start + batch_size]
            for start in range(0, len(remaining_paths), batch_size)
        ]
        loaded: queue.Queue[torch.Tensor | BaseException] = queue.Queue(
            maxsize=_PREFETCH_BATCHES
        )

        def _load_ahead() -> None:
            for batch_paths in batch_path_lists:
                try:
                    loaded.put(self._load_image_batch(batch_paths, self.image_size))
                except BaseException as exc:  # re-raised on the caller's thread below
                    loaded.put(exc)
                    return

        loader_thread = threading.Thread(target=_load_ahead, daemon=True)
        loader_thread.start()

        flush_every_batches = max(1, _CACHE_FLUSH_INTERVAL_ENTRIES // batch_size)
        is_cuda = torch.device(self.device or "cpu").type == "cuda"

        new_encodings: list[torch.Tensor] = []
        processed_new_paths: list[str] = []
        for batch_index, batch_paths in enumerate(
            tqdm(batch_path_lists, desc="Encoding images")
        ):
            batch = loaded.get()
            if isinstance(batch, BaseException):
                raise batch

            pil_images = ImageData(images=batch).get_as_pil_images()

            inputs = self.processor(
                images=pil_images, return_tensors="pt", do_rescale=False
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            outputs = self.model(**inputs)
            # detach (no grads - redundant under @torch.no_grad() but cheap
            # insurance) and move off the GPU immediately, rather than
            # letting encoded batches pile up in GPU memory until the end
            new_encodings.append(outputs.last_hidden_state.detach()[:, 0].cpu())
            processed_new_paths.extend(batch_paths)
            del inputs, outputs

            is_last_batch = batch_index == len(batch_path_lists) - 1
            if cache_file is not None and (
                is_last_batch or (batch_index + 1) % flush_every_batches == 0
            ):
                checkpoint_parts = new_encodings
                if resume_encodings is not None:
                    checkpoint_parts = [resume_encodings, *new_encodings]
                checkpoint_paths = resume_paths + processed_new_paths
                self._save_cache(
                    cache_file, torch.cat(checkpoint_parts, dim=0), checkpoint_paths
                )
                print(
                    f"{_LOG_PREFIX} checkpoint saved: {len(checkpoint_paths)}/"
                    f"{len(paths)} images -> {cache_file}"
                )
                if is_cuda:
                    torch.cuda.empty_cache()

        loader_thread.join()

        all_paths = resume_paths + processed_new_paths
        all_parts = (
            new_encodings
            if resume_encodings is None
            else [
                resume_encodings,
                *new_encodings,
            ]
        )
        all_encodings = torch.cat(all_parts, dim=0)

        if all_paths == paths:
            return all_encodings.to(self.device)

        path_to_row = {p: i for i, p in enumerate(all_paths)}
        order = torch.tensor([path_to_row[p] for p in paths], dtype=torch.long)
        return all_encodings[order].to(self.device)

    def SetDevice(self, device) -> None:
        """Moves precomputed encodings (and the fitted distribution) to the
        given device. Raw images are loaded from disk on demand in
        generate() rather than held resident, so there is nothing else to
        move.

        Args:
            device (_type_): device object/str to move to
        """

        self.device = device
        self._encodings = self._encodings.to(device)
        self.dist = Normal(
            self.dist.mean.to(device), self.dist.stddev.to(device), validate_args=None
        )

    def generate(self, data: ActionData) -> ImageData:
        """Looks up the closest precomputed encoding for each action and
        returns the corresponding real image, loaded from disk on demand.

        Args:
            data (ActionData): actions to find the nearest real image for

        Returns:
            ImageData: nearest real images, one per action
        """

        actions = data.actions.to(self._encodings.device)
        nearest = self._nearest_encoding_indices(actions)
        paths = [self._image_paths[i] for i in nearest.tolist()]

        return ImageData(
            images=self._load_image_batch(paths, self.image_size).to(self.device)
        )

    def _nearest_encoding_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns, per action, the index of its nearest row in
        `self._encodings`. Scans `self._encodings` in chunks rather than
        a single `torch.cdist(actions, self._encodings)` call, so the
        transient distance matrix stays bounded at
        `len(actions) * _NEAREST_NEIGHBOR_CHUNK_SIZE` instead of
        `len(actions) * len(self._encodings)` - relevant once
        `image_folder` holds hundreds of thousands of images.

        # ponytail: still an O(B*N*D) brute-force scan, just memory-bounded;
        # switch to an approximate/indexed nearest-neighbor library (e.g.
        # FAISS) if the per-round compute cost itself becomes the bottleneck.
        """

        def _chunk_nearest(start: int) -> tuple[torch.Tensor, torch.Tensor]:
            chunk = self._encodings[start : start + _NEAREST_NEIGHBOR_CHUNK_SIZE]
            chunk_dist, chunk_argmin = torch.cdist(actions, chunk).min(dim=1)
            return chunk_dist, chunk_argmin + start

        chunk_starts = range(0, self._encodings.shape[0], _NEAREST_NEIGHBOR_CHUNK_SIZE)
        best_dist, best_idx = _chunk_nearest(chunk_starts[0])
        for start in chunk_starts[1:]:
            chunk_dist, chunk_idx = _chunk_nearest(start)
            better = chunk_dist < best_dist
            best_dist = torch.where(better, chunk_dist, best_dist)
            best_idx = torch.where(better, chunk_idx, best_idx)

        return best_idx

    def sample_random_actions(self, N: int) -> ActionData:
        """Samples given number of vectors from the fitted encoding
        distribution.

        Args:
            N (int): noise vectors number to generate

        Raises:
            Exception: If given number of actions to generate is lower than 1

        Returns:
            ActionData: list of sampled actions
        """

        if N < 1:
            raise Exception(f"Generate noise number cannot be lower 1. Provided: {N}")

        return ActionData(actions=self.dist.sample((N,)))

    def get_input_noise_distribution(self) -> Normal:
        """Returns the Normal distribution fitted to the precomputed
        image encodings.

        Returns:
            Normal: Normal distribution object
        """

        return self.dist

    def get_media_logger(self) -> AbsMetricsLogger:
        """Returns image logger to tensorboard

        Raises:
            NotImplementedError: Not implemented

        Returns:
            AbsMetricsLogger: Image tensorboard logger
        """

        raise NotImplementedError()

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str:
        """
        return (
            "Real data gen model.\n"
            f"Image folder: {self.image_folder}\n"
            f"Encodings: {tuple(self._encodings.shape)}"
        )
