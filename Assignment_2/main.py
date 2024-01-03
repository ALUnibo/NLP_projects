from file_reader import import_features, import_labels
from dataframe_modifier import modify_stance, create_third_level_labels
from models_generator import c_model
from dataset_generator import generate_datasets
from trainer import train
from metrics import compute_metrics
from transformers import AutoTokenizer


if __name__ == '__main__':
    train_dataframe, validation_dataframe, test_dataframe = import_features()
    lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe = import_labels()
    modify_stance(train_dataframe, validation_dataframe, test_dataframe)
    training_set, validation_set, test_set = create_third_level_labels(lab_train_dataframe, lab_validation_dataframe,
                                                                       lab_test_dataframe, train_dataframe,
                                                                       validation_dataframe, test_dataframe)

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")  # bert-base-uncased
    train_data, val_data, test_data = generate_datasets(tokenizer, training_set, validation_set, test_set)

    model = c_model(len(train_data['labels'][0]))
    # train(tokenizer, model, train_data, val_data, compute_metrics)
