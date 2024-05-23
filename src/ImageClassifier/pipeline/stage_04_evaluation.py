from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.Model_evaluation import Evaluation
from ImageClassifier import logger

STAGE_NAME = "Model Evaluation"

class ModelEvaluation:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
        obj = ModelEvaluation()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<< \n\n X*****************X")
    except Exception as e:
        logger.exception(e)
        raise e