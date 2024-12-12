import pandas as pd
from sklearn.model_selection import StratifiedKFold
from dotenv import load_dotenv

load_dotenv()

path_name = "data/goodreads-books-reviews-290312/"
csv_name = "goodreads_train.csv"
fold_name = "goodreads_train_folds.csv"

if __name__ == "__main__":

    df = pd.read_csv(path_name + csv_name)
    df.loc[:, 'kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.review_text.values
    y = df.rating.values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

    for fold, (trn_, val_) in enumerate(kf.split(X,y)):
        print("TRAIN", trn_, "VAL", val_)
        df.loc[val_, "kfold"] = fold
    
    print(df.kfold.value_counts())
    df.to_csv(path_name + fold_name, index = False)