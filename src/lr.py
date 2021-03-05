# ohe_lr.py
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import joblib
import os
import config
import pickle


def run(fold):
    df = pd.read_csv('../input/train_fold.csv')
    test = pd.read_csv('../input/test.csv')

    features = [
        f for f in df.columns if f not in ('id','target','kfold')
    ]

    # replacing missing values with 'NONE'
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
        test.loc[:, col] = test[col].astype(str).fillna('NONE')

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([
        df_train[features], df_valid[features], test[features]], axis=0)

    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    test = ohe.transform(test[features])
    
    pickle.dump(test, open('./input/test_processed.pkl', 'wb'))

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)

    valid_pred = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_pred)

    print(f"Fold= {fold}, Auc = {auc}")

    # Saving the model
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"lr.bin"))


if __name__ == "__main__":
    # for fold in range(5):
    run(1)