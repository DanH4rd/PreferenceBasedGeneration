import pytest
import torch


class TestDiscriminationModel:
    def test_stackgan_discriminator(self, facade_gen_model, facade_disc_model):
        assert (
            len(
                facade_disc_model.discriminate(
                    facade_gen_model.generate(
                        facade_gen_model.sample_random_actions(N=1)
                    )
                ).shape
            )
            == 1
        )
        assert (
            facade_disc_model.discriminate(
                facade_gen_model.generate(facade_gen_model.sample_random_actions(N=1))
            ).shape[0]
            == 1
        )
        assert (
            facade_disc_model.discriminate(
                facade_gen_model.generate(facade_gen_model.sample_random_actions(5))
            ).shape[0]
            == 5
        )

    def test_set_device_moves_model_and_discriminate_still_works(
        self, facade_gen_model, facade_disc_model
    ):
        """SetDevice must actually move the model (and discriminate() must
        move the input images to match), not just leave it stuck on cpu
        (see StackGanDiscModel.SetDevice)."""

        facade_disc_model.SetDevice("cpu")

        assert facade_disc_model.device == "cpu"
        assert next(facade_disc_model.model.parameters()).device.type == "cpu"

        images = facade_gen_model.generate(facade_gen_model.sample_random_actions(2))
        scores = facade_disc_model.discriminate(images)

        assert scores.shape[0] == 2

    def test_set_device_supports_cuda_round_trip(
        self, facade_gen_model, facade_disc_model
    ):
        """Switching to CUDA and back to CPU must both work: DataParallel's
        device_ids has to be kept in sync with the target device, not just
        the underlying tensors (see StackGanDiscModel.SetDevice)."""

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in this environment")

        # facade_disc_model is a session-scoped fixture shared by other test
        # modules - always restore it to cpu, even if an assertion below fails.
        try:
            facade_disc_model.SetDevice("cuda:0")
            assert facade_disc_model.device == "cuda:0"
            assert next(facade_disc_model.model.parameters()).device.type == "cuda"

            images = facade_gen_model.generate(
                facade_gen_model.sample_random_actions(2)
            )
            scores = facade_disc_model.discriminate(images)
            assert scores.shape[0] == 2
        finally:
            facade_disc_model.SetDevice("cpu")

        assert next(facade_disc_model.model.parameters()).device.type == "cpu"

        images = facade_gen_model.generate(facade_gen_model.sample_random_actions(2))
        scores = facade_disc_model.discriminate(images)
        assert scores.shape[0] == 2
