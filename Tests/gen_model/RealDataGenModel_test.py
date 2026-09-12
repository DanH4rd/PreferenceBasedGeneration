import importlib

import pytest
import torch

from src.GenModel.RealDataGenModel import RealDataGenModel

# `import src.GenModel.RealDataGenModel as x` would bind x to the RealDataGenModel
# *class* here, not the submodule - src/GenModel/__init__.py's `from .RealDataGenModel
# import RealDataGenModel` rebinds that attribute name on the package after import
# (see CLAUDE.md's note on this exact footgun for src/DataStructures). importlib
# sidesteps the package attribute entirely and returns the actual submodule.
real_data_gen_model_module = importlib.import_module("src.GenModel.RealDataGenModel")


def _bare_model(image_size: int = 4) -> RealDataGenModel:
    """Builds a RealDataGenModel without running __init__ (which needs a
    real image folder and downloads the ViT model), so only the
    background-thread prefetch logic in _encode_images is exercised.
    """

    model = object.__new__(RealDataGenModel)
    model.image_size = image_size
    model.device = None
    return model


_FAKE_EMBED_DIM = 8


def _fake_encoder(
    model: RealDataGenModel,
    loaded_batches: list[list[str]],
    embed_dim: int = _FAKE_EMBED_DIM,
) -> None:
    def fake_load_image_batch(paths: list[str], image_size: int) -> torch.Tensor:
        loaded_batches.append(paths)
        return torch.zeros(len(paths), 3, image_size, image_size)

    def fake_processor(images, return_tensors, do_rescale):
        return {
            "pixel_values": torch.zeros(
                len(images), 3, model.image_size, model.image_size
            )
        }

    class FakeOutputs:
        def __init__(self, batch_size: int) -> None:
            self.last_hidden_state = torch.zeros(batch_size, 1, embed_dim)

    def fake_model(**kwargs) -> FakeOutputs:
        return FakeOutputs(kwargs["pixel_values"].shape[0])

    model._load_image_batch = fake_load_image_batch  # pyright: ignore[reportAttributeAccessIssue]
    model.processor = fake_processor  # pyright: ignore[reportAttributeAccessIssue]
    model.model = fake_model  # pyright: ignore[reportAttributeAccessIssue]


class TestRealDataGenModelEncodeImagesThreading:
    def test_encode_images_preserves_batch_order(self):
        """The loader thread must feed batches to the model in the same
        order they were requested, even though loading now happens
        concurrently with inference."""

        model = _bare_model()
        loaded_batches: list[list[str]] = []
        _fake_encoder(model, loaded_batches)

        paths = [f"img{i}.png" for i in range(10)]
        model._encode_images(paths, batch_size=3)

        assert loaded_batches == [paths[0:3], paths[3:6], paths[6:9], paths[9:10]]

    def test_encode_images_propagates_loader_exception(self):
        """A failure in the background loader thread (e.g. a corrupt image
        file) must surface on the caller's thread, not hang or vanish."""

        model = _bare_model()

        def failing_load_image_batch(paths: list[str], image_size: int) -> torch.Tensor:
            raise ValueError("corrupt image")

        model._load_image_batch = failing_load_image_batch  # pyright: ignore[reportAttributeAccessIssue]
        model.processor = lambda **kwargs: {}  # pyright: ignore[reportAttributeAccessIssue]
        model.model = lambda **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(ValueError, match="corrupt image"):
            model._encode_images(["img0.png"], batch_size=3)

    def test_encode_images_checkpoints_progress_with_paths_prefix(self, monkeypatch):
        """With a cache_file given, progress must be checkpointed
        periodically using only the paths encoded so far, so a run
        interrupted partway through has something to resume from."""

        monkeypatch.setattr(
            real_data_gen_model_module, "_CACHE_FLUSH_INTERVAL_ENTRIES", 2
        )

        model = _bare_model()
        _fake_encoder(model, loaded_batches=[])

        saved_paths: list[list[str]] = []

        def fake_save_cache(cache_file, encodings, paths):
            saved_paths.append(list(paths))

        model._save_cache = fake_save_cache  # pyright: ignore[reportAttributeAccessIssue]

        paths = [f"img{i}.png" for i in range(5)]
        model._encode_images(paths, batch_size=2, cache_file="fake_cache.pt")

        # batches are [0:2], [2:4], [4:5]; flush_every_batches = max(1, 2 // 2)
        # = 1, so a checkpoint is written after every batch
        assert saved_paths == [paths[0:2], paths[0:4], paths[0:5]]

    def test_get_encodings_never_flushes_without_a_cache_file(self):
        """The sanity-check re-encode of a handful of sample images must
        not attempt to checkpoint anything - there's no cache_file to
        write to, and it must not disturb the cache file under test."""

        model = _bare_model()
        _fake_encoder(model, loaded_batches=[])

        model._save_cache = lambda *args, **kwargs: pytest.fail(  # pyright: ignore[reportAttributeAccessIssue]
            "must not save a cache when cache_file is not given"
        )

        paths = [f"img{i}.png" for i in range(5)]
        model._encode_images(paths, batch_size=2)

    def test_encode_images_resumes_from_partial_cache(self):
        """resume_cache must be treated as already-done: only the paths
        not present in it get loaded/encoded, and the resumed rows are
        carried through unchanged into the result."""

        model = _bare_model()
        loaded_batches: list[list[str]] = []
        _fake_encoder(model, loaded_batches)

        paths = [f"img{i}.png" for i in range(6)]
        resume_encodings = torch.arange(
            2 * _FAKE_EMBED_DIM, dtype=torch.float32
        ).reshape(2, _FAKE_EMBED_DIM)
        resume_cache = {"paths": paths[:2], "encodings": resume_encodings}

        result = model._encode_images(paths, batch_size=2, resume_cache=resume_cache)

        # only the paths not covered by the resume cache should be loaded
        assert loaded_batches == [paths[2:4], paths[4:6]]
        assert result.shape == (6, _FAKE_EMBED_DIM)
        assert torch.equal(result[:2], resume_encodings)

    def test_encode_images_resume_is_order_resistant(self):
        """A resume cache's paths are matched by identity, not position -
        a cache that's out of order and covers a scattered subset (not a
        prefix) of the current image list must still be used correctly,
        and the result must come back in the same order as `paths`."""

        model = _bare_model()
        loaded_batches: list[list[str]] = []
        _fake_encoder(model, loaded_batches)

        paths = ["img0.png", "img1.png", "img2.png", "img3.png"]
        # covers img2 and img0, out of order, not a prefix of `paths`
        resume_cache = {
            "paths": ["img2.png", "img0.png"],
            "encodings": torch.tensor(
                [[20.0] * _FAKE_EMBED_DIM, [0.0] * _FAKE_EMBED_DIM]
            ),
        }

        result = model._encode_images(paths, batch_size=2, resume_cache=resume_cache)

        # only the two paths not covered by the resume cache get loaded
        assert loaded_batches == [["img1.png", "img3.png"]]
        # result rows line up with `paths` order regardless of the
        # scattered/out-of-order resume cache
        assert torch.equal(result[0], resume_cache["encodings"][1])  # img0
        assert torch.equal(result[2], resume_cache["encodings"][0])  # img2

    def test_encode_images_resume_checkpoints_use_cumulative_paths(self, monkeypatch):
        """A checkpoint written while resuming must record every path
        accounted for so far (already-resumed + newly encoded), not just
        the paths encoded in this call."""

        monkeypatch.setattr(
            real_data_gen_model_module, "_CACHE_FLUSH_INTERVAL_ENTRIES", 2
        )

        model = _bare_model()
        _fake_encoder(model, loaded_batches=[])

        saved_paths: list[list[str]] = []
        model._save_cache = lambda cache_file, encodings, paths: saved_paths.append(  # pyright: ignore[reportAttributeAccessIssue]
            list(paths)
        )

        paths = [f"img{i}.png" for i in range(6)]
        resume_cache = {
            "paths": paths[:2],
            "encodings": torch.zeros(2, _FAKE_EMBED_DIM),
        }

        model._encode_images(
            paths,
            batch_size=2,
            cache_file="fake_cache.pt",
            resume_cache=resume_cache,
        )

        # already 2 accounted for; batches of 2 remain: [2:4], [4:6]; each flushes
        assert saved_paths == [paths[0:4], paths[0:6]]


class TestRealDataGenModelCachePassesSanityCheck:
    """_cache_passes_sanity_check compares by cosine similarity, not raw
    torch.allclose - re-running the same images through the model on CPU
    doesn't reproduce bit-identical activations (thread-scheduling changes
    floating-point summation order across a 12-layer transformer), so an
    absolute tolerance on raw hidden states was flagging harmless numeric
    noise as a stale cache."""

    def test_tolerates_floating_point_noise(self):
        model = _bare_model()
        paths = ["a", "b", "c"]
        cached_encodings = torch.randn(3, _FAKE_EMBED_DIM)
        cached = {"paths": paths, "encodings": cached_encodings}

        # noise 3 orders of magnitude smaller than the signal - the kind of
        # run-to-run float jitter a real re-encode would actually produce
        noisy_by_path = {
            p: cached_encodings[i] + torch.randn(_FAKE_EMBED_DIM) * 1e-3
            for i, p in enumerate(paths)
        }
        model._encode_images = lambda sample_paths, batch_size: torch.stack(  # pyright: ignore[reportAttributeAccessIssue]
            [noisy_by_path[p] for p in sample_paths]
        )

        assert model._cache_passes_sanity_check(cached, batch_size=2) is True

    def test_rejects_genuinely_different_embeddings(self):
        """A cache holding the wrong embeddings entirely (not just noisy
        ones) must still be rejected."""

        model = _bare_model()
        paths = ["a", "b", "c"]
        cached_encodings = torch.eye(3, _FAKE_EMBED_DIM)
        cached = {"paths": paths, "encodings": cached_encodings}

        # negated: cosine similarity -1, as wrong as two vectors can be
        unrelated_by_path = {p: -cached_encodings[i] for i, p in enumerate(paths)}
        model._encode_images = lambda sample_paths, batch_size: torch.stack(  # pyright: ignore[reportAttributeAccessIssue]
            [unrelated_by_path[p] for p in sample_paths]
        )

        assert model._cache_passes_sanity_check(cached, batch_size=2) is False


class TestRealDataGenModelLoadCache:
    """_load_cache no longer compares cached paths against
    self._image_paths at all (see _encode_images for the identity-based
    reconciliation that makes resume order/subset resistant) - it only
    checks the cache's own internal consistency and metadata."""

    def _write_cache(
        self,
        tmp_path,
        paths: list[str],
        num_encodings: int,
        image_size: int = 4,
        encoder_name: str | None = None,
    ) -> str:
        cache_file = str(tmp_path / "cache.pt")
        torch.save(
            {
                "encodings": torch.zeros(num_encodings, _FAKE_EMBED_DIM),
                "paths": paths,
                "image_size": image_size,
                "encoder_name": encoder_name
                or real_data_gen_model_module._ENCODER_NAME,
            },
            cache_file,
        )
        return cache_file

    def test_accepts_complete_cache(self, tmp_path):
        model = _bare_model()
        cache_file = self._write_cache(tmp_path, paths=["a", "b", "c"], num_encodings=3)

        cached = model._load_cache(cache_file)

        assert cached is not None
        assert cached["paths"] == ["a", "b", "c"]

    def test_accepts_partial_cache_regardless_of_order_or_coverage(self, tmp_path):
        """A cache covering only a scattered, out-of-order subset of the
        current images must still be accepted - _encode_images decides
        what's reusable by path identity, not _load_cache."""

        model = _bare_model()
        cache_file = self._write_cache(tmp_path, paths=["c", "a"], num_encodings=2)

        cached = model._load_cache(cache_file)

        assert cached is not None
        assert cached["paths"] == ["c", "a"]

    def test_rejects_mismatched_encoding_count(self, tmp_path):
        """A corrupted/truncated checkpoint (encodings count doesn't match
        its own paths list) must not be trusted at all."""

        model = _bare_model()
        cache_file = self._write_cache(tmp_path, paths=["a", "b"], num_encodings=1)

        assert model._load_cache(cache_file) is None

    def test_rejects_mismatched_image_size(self, tmp_path):
        model = _bare_model(image_size=4)
        cache_file = self._write_cache(
            tmp_path, paths=["a", "b"], num_encodings=2, image_size=999
        )

        assert model._load_cache(cache_file) is None
