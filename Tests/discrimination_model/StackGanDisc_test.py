from src.DiscModel.StackGanDiscModel import StackGanDiscModel
from src.GenModel.StackGanGenModel import StackGanGenModel


class TestDiscriminationModel:
    def test_stackgan_discriminator(self):
        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )

        disc_model = StackGanDiscModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netD2.pth",
            scale_level=2,
        )

        assert (
            len(
                disc_model.discriminate(
                    gen_model.generate(gen_model.sample_random_actions(N=1))
                ).shape
            )
            == 1
        )
        assert (
            disc_model.discriminate(
                gen_model.generate(gen_model.sample_random_actions(N=1))
            ).shape[0]
            == 1
        )
        assert (
            disc_model.discriminate(
                gen_model.generate(gen_model.sample_random_actions(5))
            ).shape[0]
            == 5
        )
