from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__ == "__main__":
    raw_path = "artifacts/raw/data.csv"
    processed_path = "artifacts/processed"
    model_path = "artifacts/model"
    processor = DataProcessing(raw_path=raw_path, processed_path=processed_path)

    processor.process_data()

    
    trainer = ModelTraining(processed_path=processed_path, model_path=model_path)
    trainer.run()