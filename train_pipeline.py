from config import config
from pipeline import titanic_pipe
from preprocessing.data_management import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name = config.TRAIN_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.KEEP_FEATURES],  # predictors
        data[config.TARGET],
        test_size=config.SPLIT_FRAC,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.RANDOM_STATE,
    )

    # fit model
    titanic_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_save = titanic_pipe)


if __name__ == "__main__":
    run_training()
