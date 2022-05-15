# %%
# Setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# %%
def load_and_process(path:str="00_data/fifa_cleaned.csv")-> pd.DataFrame:
    # Data loading
    df_raw = pd.read_csv(path,index_col=False)
    # Subsetting
    cols_to_keep = ["age",
                    "height_cm",
                    "weight_kgs",
                    "positions",
                    "overall_rating",
                    "weak_foot(1-5)",
                    "skill_moves(1-5)",
                    "crossing",
                    "finishing",
                    "heading_accuracy",
                    "short_passing",
                    "volleys",
                    "dribbling",
                    "curve",
                    "freekick_accuracy",
                    "long_passing",
                    "ball_control",
                    "acceleration",
                    "sprint_speed",
                    "agility",
                    "reactions",
                    "balance",
                    "shot_power",
                    "jumping",
                    "stamina",
                    "strength",
                    "long_shots",
                    "aggression",
                    "interceptions",
                    "positioning",
                    "vision",
                    "penalties",
                    "composure",
                    "marking",
                    "standing_tackle",
                    "sliding_tackle"
                    ]
    df = df_raw[cols_to_keep]
    # Unnest positions
    df = df.assign(positions=df.positions.str.split(",")).explode("positions").copy()
    # Scaling
    positions = df["positions"].values
    array_to_scale = df.drop(columns=["positions"]).to_numpy()
    scaler = StandardScaler().fit(array_to_scale)
    scaled_array = scaler.transform(array_to_scale)
    # Combine
    df_pred = pd.DataFrame(positions,columns=["positions"])
    cols = cols_to_keep.copy()
    cols.remove("positions")
    df_temp = pd.DataFrame(scaled_array,columns=cols)
    df_pred = pd.concat([df_pred,df_temp],axis=1)
    # Delete Goalkeepers
    df_pred = df_pred.query("positions!='GK'")
    # One Hot Encoding
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df_pred[['positions']])
    df_transformed = pd.DataFrame(transformed.toarray(),columns=list(ohe.categories_[0]))
    df_pred = df_pred.drop(columns=["positions"])
    df_transformed = df_transformed.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    df_pred = pd.concat([df_pred,df_transformed],axis=1)
    # Train Test Split
    df_train, df_valid = train_test_split(df_pred,train_size=0.8,random_state=42,shuffle=True)
    return(df_train, df_valid)