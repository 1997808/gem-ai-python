# Import dependencies
import pandas as pd
import numpy as np
import sklearn.externals as extjoblib
from sklearn import preprocessing
from sklearn import utils
import joblib

# Load the dataset in a dataframe object and include only four features as mentioned
url = "http://localhost:5000/api/train-data"
file = '23000.csv'
df = pd.read_csv(file)
columnNames = df.columns.tolist()
# include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
include = [
   'bot_0_playerId',
   'bot_0_id',
   'bot_0_attack',
   'bot_0_hp',
   'bot_0_mana',
   'bot_0_maxMana',
   'bot_0_gemTypes_0',
   'bot_0_gemTypes_1',
   'bot_0_gems_0',
   'bot_0_gems_1',
   'bot_1_playerId',
   'bot_1_id',
   'bot_1_attack',
   'bot_1_hp',
   'bot_1_mana',
   'bot_1_maxMana',
   'bot_1_gemTypes_0',
   'bot_1_gemTypes_1',
   'bot_1_gems_0',
   'bot_1_gems_1',
   'bot_2_playerId',
   'bot_2_id',
   'bot_2_attack',
   'bot_2_hp',
   'bot_2_mana',
   'bot_2_maxMana',
   'bot_2_gemTypes_0',
   'bot_2_gemTypes_1',
   'bot_2_gems_0',
   'bot_2_gems_1',
   'currentBoard_0_signature',
   'currentBoard_0_index',
   'currentBoard_0_type',
   'currentBoard_0_modifier',
   'currentBoard_0_x',
   'currentBoard_0_y',
   'currentBoard_1_signature',
   'currentBoard_1_index',
   'currentBoard_1_type',
   'currentBoard_1_modifier',
   'currentBoard_1_x',
   'currentBoard_1_y',
   'currentBoard_2_signature',
   'currentBoard_2_index',
   'currentBoard_2_type',
   'currentBoard_2_modifier',
   'currentBoard_2_x',
   'currentBoard_2_y',
   'currentBoard_3_signature',
   'currentBoard_3_index',
   'currentBoard_3_type',
   'currentBoard_3_modifier',
   'currentBoard_3_x',
   'currentBoard_3_y',
   'currentBoard_4_signature',
   'currentBoard_4_index',
   'currentBoard_4_type',
   'currentBoard_4_modifier',
   'currentBoard_4_x',
   'currentBoard_4_y',
   'currentBoard_5_signature',
   'currentBoard_5_index',
   'currentBoard_5_type',
   'currentBoard_5_modifier',
   'currentBoard_5_x',
   'currentBoard_5_y',
   'currentBoard_6_signature',
   'currentBoard_6_index',
   'currentBoard_6_type',
   'currentBoard_6_modifier',
   'currentBoard_6_x',
   'currentBoard_6_y',
   'currentBoard_7_signature',
   'currentBoard_7_index',
   'currentBoard_7_type',
   'currentBoard_7_modifier',
   'currentBoard_7_x',
   'currentBoard_7_y',
   'currentBoard_8_signature',
   'currentBoard_8_index',
   'currentBoard_8_type',
   'currentBoard_8_modifier',
   'currentBoard_8_x',
   'currentBoard_8_y',
   'currentBoard_9_signature',
   'currentBoard_9_index',
   'currentBoard_9_type',
   'currentBoard_9_modifier',
   'currentBoard_9_x',
   'currentBoard_9_y',
   'currentBoard_10_signature',
   'currentBoard_10_index',
   'currentBoard_10_type',
   'currentBoard_10_modifier',
   'currentBoard_10_x',
   'currentBoard_10_y',
   'currentBoard_11_signature',
   'currentBoard_11_index',
   'currentBoard_11_type',
   'currentBoard_11_modifier',
   'currentBoard_11_x',
   'currentBoard_11_y',
   'currentBoard_12_signature',
   'currentBoard_12_index',
   'currentBoard_12_type',
   'currentBoard_12_modifier',
   'currentBoard_12_x',
   'currentBoard_12_y',
   'currentBoard_13_signature',
   'currentBoard_13_index',
   'currentBoard_13_type',
   'currentBoard_13_modifier',
   'currentBoard_13_x',
   'currentBoard_13_y',
   'currentBoard_14_signature',
   'currentBoard_14_index',
   'currentBoard_14_type',
   'currentBoard_14_modifier',
   'currentBoard_14_x',
   'currentBoard_14_y',
   'currentBoard_15_signature',
   'currentBoard_15_index',
   'currentBoard_15_type',
   'currentBoard_15_modifier',
   'currentBoard_15_x',
   'currentBoard_15_y',
   'currentBoard_16_signature',
   'currentBoard_16_index',
   'currentBoard_16_type',
   'currentBoard_16_modifier',
   'currentBoard_16_x',
   'currentBoard_16_y',
   'currentBoard_17_signature',
   'currentBoard_17_index',
   'currentBoard_17_type',
   'currentBoard_17_modifier',
   'currentBoard_17_x',
   'currentBoard_17_y',
   'currentBoard_18_signature',
   'currentBoard_18_index',
   'currentBoard_18_type',
   'currentBoard_18_modifier',
   'currentBoard_18_x',
   'currentBoard_18_y',
   'currentBoard_19_signature',
   'currentBoard_19_index',
   'currentBoard_19_type',
   'currentBoard_19_modifier',
   'currentBoard_19_x',
   'currentBoard_19_y',
   'currentBoard_20_signature',
   'currentBoard_20_index',
   'currentBoard_20_type',
   'currentBoard_20_modifier',
   'currentBoard_20_x',
   'currentBoard_20_y',
   'currentBoard_21_signature',
   'currentBoard_21_index',
   'currentBoard_21_type',
   'currentBoard_21_modifier',
   'currentBoard_21_x',
   'currentBoard_21_y',
   'currentBoard_22_signature',
   'currentBoard_22_index',
   'currentBoard_22_type',
   'currentBoard_22_modifier',
   'currentBoard_22_x',
   'currentBoard_22_y',
   'currentBoard_23_signature',
   'currentBoard_23_index',
   'currentBoard_23_type',
   'currentBoard_23_modifier',
   'currentBoard_23_x',
   'currentBoard_23_y',
   'currentBoard_24_signature',
   'currentBoard_24_index',
   'currentBoard_24_type',
   'currentBoard_24_modifier',
   'currentBoard_24_x',
   'currentBoard_24_y',
   'currentBoard_25_signature',
   'currentBoard_25_index',
   'currentBoard_25_type',
   'currentBoard_25_modifier',
   'currentBoard_25_x',
   'currentBoard_25_y',
   'currentBoard_26_signature',
   'currentBoard_26_index',
   'currentBoard_26_type',
   'currentBoard_26_modifier',
   'currentBoard_26_x',
   'currentBoard_26_y',
   'currentBoard_27_signature',
   'currentBoard_27_index',
   'currentBoard_27_type',
   'currentBoard_27_modifier',
   'currentBoard_27_x',
   'currentBoard_27_y',
   'currentBoard_28_signature',
   'currentBoard_28_index',
   'currentBoard_28_type',
   'currentBoard_28_modifier',
   'currentBoard_28_x',
   'currentBoard_28_y',
   'currentBoard_29_signature',
   'currentBoard_29_index',
   'currentBoard_29_type',
   'currentBoard_29_modifier',
   'currentBoard_29_x',
   'currentBoard_29_y',
   'currentBoard_30_signature',
   'currentBoard_30_index',
   'currentBoard_30_type',
   'currentBoard_30_modifier',
   'currentBoard_30_x',
   'currentBoard_30_y',
   'currentBoard_31_signature',
   'currentBoard_31_index',
   'currentBoard_31_type',
   'currentBoard_31_modifier',
   'currentBoard_31_x',
   'currentBoard_31_y',
   'currentBoard_32_signature',
   'currentBoard_32_index',
   'currentBoard_32_type',
   'currentBoard_32_modifier',
   'currentBoard_32_x',
   'currentBoard_32_y',
   'currentBoard_33_signature',
   'currentBoard_33_index',
   'currentBoard_33_type',
   'currentBoard_33_modifier',
   'currentBoard_33_x',
   'currentBoard_33_y',
   'currentBoard_34_signature',
   'currentBoard_34_index',
   'currentBoard_34_type',
   'currentBoard_34_modifier',
   'currentBoard_34_x',
   'currentBoard_34_y',
   'currentBoard_35_signature',
   'currentBoard_35_index',
   'currentBoard_35_type',
   'currentBoard_35_modifier',
   'currentBoard_35_x',
   'currentBoard_35_y',
   'currentBoard_36_signature',
   'currentBoard_36_index',
   'currentBoard_36_type',
   'currentBoard_36_modifier',
   'currentBoard_36_x',
   'currentBoard_36_y',
   'currentBoard_37_signature',
   'currentBoard_37_index',
   'currentBoard_37_type',
   'currentBoard_37_modifier',
   'currentBoard_37_x',
   'currentBoard_37_y',
   'currentBoard_38_signature',
   'currentBoard_38_index',
   'currentBoard_38_type',
   'currentBoard_38_modifier',
   'currentBoard_38_x',
   'currentBoard_38_y',
   'currentBoard_39_signature',
   'currentBoard_39_index',
   'currentBoard_39_type',
   'currentBoard_39_modifier',
   'currentBoard_39_x',
   'currentBoard_39_y',
   'currentBoard_40_signature',
   'currentBoard_40_index',
   'currentBoard_40_type',
   'currentBoard_40_modifier',
   'currentBoard_40_x',
   'currentBoard_40_y',
   'currentBoard_41_signature',
   'currentBoard_41_index',
   'currentBoard_41_type',
   'currentBoard_41_modifier',
   'currentBoard_41_x',
   'currentBoard_41_y',
   'currentBoard_42_signature',
   'currentBoard_42_index',
   'currentBoard_42_type',
   'currentBoard_42_modifier',
   'currentBoard_42_x',
   'currentBoard_42_y',
   'currentBoard_43_signature',
   'currentBoard_43_index',
   'currentBoard_43_type',
   'currentBoard_43_modifier',
   'currentBoard_43_x',
   'currentBoard_43_y',
   'currentBoard_44_signature',
   'currentBoard_44_index',
   'currentBoard_44_type',
   'currentBoard_44_modifier',
   'currentBoard_44_x',
   'currentBoard_44_y',
   'currentBoard_45_signature',
   'currentBoard_45_index',
   'currentBoard_45_type',
   'currentBoard_45_modifier',
   'currentBoard_45_x',
   'currentBoard_45_y',
   'currentBoard_46_signature',
   'currentBoard_46_index',
   'currentBoard_46_type',
   'currentBoard_46_modifier',
   'currentBoard_46_x',
   'currentBoard_46_y',
   'currentBoard_47_signature',
   'currentBoard_47_index',
   'currentBoard_47_type',
   'currentBoard_47_modifier',
   'currentBoard_47_x',
   'currentBoard_47_y',
   'currentBoard_48_signature',
   'currentBoard_48_index',
   'currentBoard_48_type',
   'currentBoard_48_modifier',
   'currentBoard_48_x',
   'currentBoard_48_y',
   'currentBoard_49_signature',
   'currentBoard_49_index',
   'currentBoard_49_type',
   'currentBoard_49_modifier',
   'currentBoard_49_x',
   'currentBoard_49_y',
   'currentBoard_50_signature',
   'currentBoard_50_index',
   'currentBoard_50_type',
   'currentBoard_50_modifier',
   'currentBoard_50_x',
   'currentBoard_50_y',
   'currentBoard_51_signature',
   'currentBoard_51_index',
   'currentBoard_51_type',
   'currentBoard_51_modifier',
   'currentBoard_51_x',
   'currentBoard_51_y',
   'currentBoard_52_signature',
   'currentBoard_52_index',
   'currentBoard_52_type',
   'currentBoard_52_modifier',
   'currentBoard_52_x',
   'currentBoard_52_y',
   'currentBoard_53_signature',
   'currentBoard_53_index',
   'currentBoard_53_type',
   'currentBoard_53_modifier',
   'currentBoard_53_x',
   'currentBoard_53_y',
   'currentBoard_54_signature',
   'currentBoard_54_index',
   'currentBoard_54_type',
   'currentBoard_54_modifier',
   'currentBoard_54_x',
   'currentBoard_54_y',
   'currentBoard_55_signature',
   'currentBoard_55_index',
   'currentBoard_55_type',
   'currentBoard_55_modifier',
   'currentBoard_55_x',
   'currentBoard_55_y',
   'currentBoard_56_signature',
   'currentBoard_56_index',
   'currentBoard_56_type',
   'currentBoard_56_modifier',
   'currentBoard_56_x',
   'currentBoard_56_y',
   'currentBoard_57_signature',
   'currentBoard_57_index',
   'currentBoard_57_type',
   'currentBoard_57_modifier',
   'currentBoard_57_x',
   'currentBoard_57_y',
   'currentBoard_58_signature',
   'currentBoard_58_index',
   'currentBoard_58_type',
   'currentBoard_58_modifier',
   'currentBoard_58_x',
   'currentBoard_58_y',
   'currentBoard_59_signature',
   'currentBoard_59_index',
   'currentBoard_59_type',
   'currentBoard_59_modifier',
   'currentBoard_59_x',
   'currentBoard_59_y',
   'currentBoard_60_signature',
   'currentBoard_60_index',
   'currentBoard_60_type',
   'currentBoard_60_modifier',
   'currentBoard_60_x',
   'currentBoard_60_y',
   'currentBoard_61_signature',
   'currentBoard_61_index',
   'currentBoard_61_type',
   'currentBoard_61_modifier',
   'currentBoard_61_x',
   'currentBoard_61_y',
   'currentBoard_62_signature',
   'currentBoard_62_index',
   'currentBoard_62_type',
   'currentBoard_62_modifier',
   'currentBoard_62_x',
   'currentBoard_62_y',
   'currentBoard_63_signature',
   'currentBoard_63_index',
   'currentBoard_63_type',
   'currentBoard_63_modifier',
   'currentBoard_63_x',
   'currentBoard_63_y',
   'enemy_0_playerId',
   'enemy_0_id',
   'enemy_0_attack',
   'enemy_0_hp',
   'enemy_0_mana',
   'enemy_0_maxMana',
   'enemy_0_gemTypes_0',
   'enemy_0_gemTypes_1',
   'enemy_0_gems_0',
   'enemy_0_gems_1',
   'enemy_1_playerId',
   'enemy_1_id',
   'enemy_1_attack',
   'enemy_1_hp',
   'enemy_1_mana',
   'enemy_1_maxMana',
   'enemy_1_gemTypes_0',
   'enemy_1_gemTypes_1',
   'enemy_1_gems_0',
   'enemy_1_gems_1',
   'enemy_2_playerId',
   'enemy_2_id',
   'enemy_2_attack',
   'enemy_2_hp',
   'enemy_2_mana',
   'enemy_2_maxMana',
   'enemy_2_gemTypes_0',
   'enemy_2_gemTypes_1',
   'enemy_2_gems_0',
   'enemy_2_gems_1',
   'matchGem_index1',
   'matchGem_index2',
   'matchGem_type',
   'label',
]

df_ = df[columnNames]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'label'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
lr.fit(x, y_transformed)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")