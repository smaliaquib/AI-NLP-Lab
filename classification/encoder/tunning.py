import optuna
from your_model import train, evaluate  # Import the train and evaluate functions from your model

def objective(trial):
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-4)
    MAX_LEN = trial.suggest_categorical("MAX_LEN", [512])
    VALID_BATCH_SIZE = trial.suggest_int("VALID_BATCH_SIZE", 2, 16)
    EPOCHS = trial.suggest_int("EPOCHS", 1, 10)

    # Use the train and evaluate functions from your model
    train_dataset = TextClassification()
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False
    )
    model = YourModel()  # Initialize your model

    accuracy = evaluate(trial, train_dataset, valid_loader, model)
    return -accuracy  # Optuna minimizes the objective function, so we negate the accuracy

def tune():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), study_name='stumble')
    study.optimize(objective, n_trials=10)

    # Get the best hyperparameters
    best_trial = study.best_trial
    LEARNING_RATE = best_trial.params["learning_rate"]
    MAX_LEN = best_trial.params["MAX_LEN"]
    VALID_BATCH_SIZE = best_trial.params["VALID_BATCH_SIZE"]

    print(f"Best hyperparameters: LR={LEARNING_RATE}, MAX_LEN={MAX_LEN}, VALID_BATCH_SIZE={VALID_BATCH_SIZE}")

if __name__ == "__main__":
    tune()