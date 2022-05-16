# %%
# Setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.preprocessing import load_and_process
# %%
# Train Test Split
df_train, df_valid = load_and_process(path="00_data/fifa_cleaned.csv")

## Train
y_train = df_train["overall_rating"].values
x_train = df_train.drop(columns=["overall_rating"]).values
## Test
y_test = df_valid["overall_rating"].values
x_test = df_valid.drop(columns=["overall_rating"]).values

# %%
# Training Loop
loop_list = []
## Define Training Loop
for i in tqdm(range(1,6)):
    ### Build Regressors
    regr_1 = GradientBoostingRegressor(learning_rate=0.1,max_depth=i,min_samples_split=10,min_samples_leaf=5,n_estimators=100,random_state=42)
    ### Fit Regressors
    regr_1.fit(x_train, y_train)
    ### Train Scores
    train_score = round(regr_1.score(x_train, y_train),3)
    ### Test Scores
    test_score = round(regr_1.score(x_test, y_test),3)
    loop_dict = {
        "max_depth": i,
        "train_score": train_score,
        "test_score": test_score
    }
    loop_list.append(loop_dict)
# %%
# Choose Best Model
df_scores = pd.DataFrame(loop_list)
max_train = df_scores["test_score"].max()
df_scores = df_scores.query("test_score=="+str(max_train)).reset_index(drop=True).iloc[0]
### Build Regressors
regr_1 = GradientBoostingRegressor(learning_rate=0.1,max_depth=int(df_scores["max_depth"]),min_samples_split=10,min_samples_leaf=5,random_state=42)
### Fit Regressors
regr_1.fit(x_train, y_train)
# %%
# Summary
### Test Scores
display("Best test score: ",round(regr_1.score(x_test, y_test),3))
display("max_depth: ",int(df_scores["max_depth"]))
# %%