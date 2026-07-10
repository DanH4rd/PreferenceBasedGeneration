import pytest
import torch


class TestGenModel:
    def test_set_device_moves_model_and_generate_still_works(self, facade_gen_model):
        """SetDevice must actually move the model (and generate() must move
        the input actions to match), not just warn and no-op (see
        StackGanGenModel.SetDevice)."""

        gen_model = facade_gen_model

        gen_model.SetDevice("cpu")

        assert gen_model.device == "cpu"
        assert next(gen_model.model.parameters()).device.type == "cpu"

        actions = gen_model.sample_random_actions(N=2)
        images = gen_model.generate(actions)

        assert images.images.shape[0] == 2

    def test_set_device_supports_cuda_round_trip(self, facade_gen_model):
        """Switching to CUDA and back to CPU must both work: DataParallel's
        device_ids has to be kept in sync with the target device, not just
        the underlying tensors (see StackGanGenModel.SetDevice)."""

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in this environment")

        gen_model = facade_gen_model

        # facade_gen_model is a session-scoped fixture shared by other test
        # modules - always restore it to cpu, even if an assertion below fails.
        try:
            gen_model.SetDevice("cuda:0")
            assert gen_model.device == "cuda:0"
            assert next(gen_model.model.parameters()).device.type == "cuda"

            actions = gen_model.sample_random_actions(N=2)
            images = gen_model.generate(actions)
            assert images.images.shape[0] == 2
        finally:
            gen_model.SetDevice("cpu")

        assert next(gen_model.model.parameters()).device.type == "cpu"

        images = gen_model.generate(gen_model.sample_random_actions(N=2))
        assert images.images.shape[0] == 2
