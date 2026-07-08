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
