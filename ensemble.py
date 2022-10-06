# ライブラリのインポート
import pandas as pd
import numpy as np
from scipy import stats


# read dataframe
sleek_0 = pd.read_csv('submit/submission_sleek-shape-45-0.tsv', sep='\t', header=None)
sleek_4 = pd.read_csv('submit/submission_sleek-shape-45-4.tsv', sep='\t', header=None)
visionary = pd.read_csv('submit/submission_visionary-meadow-49.tsv', sep='\t', header=None)
young = pd.read_csv('submit/submission_young-donkey-17.tsv', sep='\t', header=None)
zesty = pd.read_csv('submit/submission_zesty-voice-48.tsv', sep='\t', header=None)

# concat dataframe
df = pd.concat(
    [sleek_0[1], sleek_4[1], visionary[1], young[1], zesty[1]],
    axis=1
)

# ensemble
ensemble_array = np.array(df).T
pred = stats.mode(ensemble_array)[0].T

# submit
test = pd.read_csv('input/sample_submit.tsv', sep='\t', header=None)
test[1] = pred
test.to_csv('submit/submission_ensemble.tsv', sep='\t', header=None, index=None)