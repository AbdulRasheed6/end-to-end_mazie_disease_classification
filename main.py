from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

STAGE_NAME= "Data Ingestion Stage"


try:

    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj= DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx============x")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME= "Prepare Base Model Stage"
try:
    logger.info(f"********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME= "Training"


try:
    logger.info(f"********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME= "Evaluation"


try:
    logger.info(f"********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e
