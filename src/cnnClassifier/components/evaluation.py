import tf_keras as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
from cnnClassifier.utils.common import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config=config


    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split= 0.30
        )

        dataflow_kwargs= dict(
            target_size= self.config.params_image_size[:2],
            batch_size= self.config.params_batch_size,
            interpolation= "bilinear"
        )

        valid_datagenerator= tf.preprocessing.image.ImageDataGenerator(
             **datagenerator_kwargs
        )

        self.valid_generator= valid_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset= "validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.Model:
        return tf.models.load_model(path)

    def evaluation(self):

        model= self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score= model.evaluate(self.valid_generator)


    def save_score(self):
        scores= {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path= Path("scores.json"), data=scores)

