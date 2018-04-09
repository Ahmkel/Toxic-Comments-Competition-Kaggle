from data_loader.data_generator import DataGenerator
from models.simple_conv_model import SimpleConvModel
from trainers.simple_conv_model_trainer import SimpleConvModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    print('Create the data generator.')
    data_generator = DataGenerator(config)

    print('Create the model.')
    model = SimpleConvModel(config, data_generator.get_word_index())

    print('Create the trainer')
    trainer = SimpleConvModelTrainer(model.model, data_generator.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Visualize the losses')
    trainer.visualize()


if __name__ == '__main__':
    main()
