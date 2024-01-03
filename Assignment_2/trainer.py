from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer


def train(tokenizer, model, train_data, val_data, compute_metrics):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="test_dir",  # where to save model
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # accelerate defines distributed training
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",  # when to report evaluation metrics/losses
        save_strategy="epoch",  # when to save checkpoint
        load_best_model_at_end=True,
        report_to='none',  # disabling wandb (default)
        label_names=['OC', 'ST', 'SE', 'CN']
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def train_torch_model():
    pass