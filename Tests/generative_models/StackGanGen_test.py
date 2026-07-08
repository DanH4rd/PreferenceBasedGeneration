from src.DataStructures import ActionData


class TestGenerativeModel:
    def test_base(self, facade_gen_model):
        model = facade_gen_model

        assert isinstance(model.sample_random_actions(N=1), ActionData)
        assert model.sample_random_actions(N=1).actions.shape[0] == 1
        assert model.sample_random_actions(N=4).actions.shape[0] == 4

        assert model.generate(model.sample_random_actions(N=1)).images.shape[0] == 1
        assert model.generate(model.sample_random_actions(N=1)).images.shape[1] == 3
        assert model.generate(model.sample_random_actions(N=5)).images.shape[0] == 5
