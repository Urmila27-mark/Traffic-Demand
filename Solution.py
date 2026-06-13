import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print('Train shape:', train.shape)
print('Test  shape:', test.shape)
print()
print('Null counts (train):')
print(train.isnull().sum())
print()
print('Demand stats:')
print(train['demand'].describe())

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df, train_stats=None):
    df = df.copy()

    # Timestamp → hour, minute, time_slot (0–95, one per 15 min)
    df['hour']      = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute']    = df['timestamp'].str.split(':').str[1].astype(int)
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15

    # Cyclical encoding of time (captures daily periodicity)
    df['sin_time'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['cos_time'] = np.cos(2 * np.pi * df['time_slot'] / 96)

    # Geohash hierarchical prefixes
    df['geo_3'] = df['geohash'].str[:3]
    df['geo_4'] = df['geohash'].str[:4]
    df['geo_5'] = df['geohash'].str[:5]

    # Binary / ordinal encodings
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(int)

    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc']  = df['Weather'].map(weather_map).fillna(-1)

    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(-1)

    # Temperature: impute with train median
    temp_med = train_stats['temp_median'] if train_stats else df['Temperature'].median()
    df['Temperature'] = df['Temperature'].fillna(temp_med)

    return df


train = preprocess(train)
test  = preprocess(test, {'temp_median': train['Temperature'].median()})
print('Preprocessing done.')

# ── Geohash label encoding (fit on combined train+test vocab) ──────────────────
combined_geo = pd.concat(
    [train[['geohash', 'geo_3', 'geo_4', 'geo_5']],
     test [['geohash', 'geo_3', 'geo_4', 'geo_5']]],
    axis=0
).reset_index(drop=True)

for col in ['geo_3', 'geo_4', 'geo_5', 'geohash']:
    le = LabelEncoder()
    le.fit(combined_geo[col])
    train[col + '_enc'] = le.transform(train[col])
    test [col + '_enc'] = le.transform(test[col])

print('Geohash encoding done.')

# ── Target encoding (mean demand per grouping key) ─────────────────────────────
global_mean = train['demand'].mean()

geo_mean    = train.groupby('geohash')['demand'].mean()
geo4_mean   = train.groupby('geo_4')['demand'].mean()
geo5_mean   = train.groupby('geo_5')['demand'].mean()
ts_mean     = train.groupby('time_slot')['demand'].mean()
geo_ts_mean = train.groupby(['geohash', 'time_slot'])['demand'].mean()

# Apply to train
train['geo_demand_mean']  = train['geohash'].map(geo_mean)
train['geo4_demand_mean'] = train['geo_4'].map(geo4_mean)
train['geo5_demand_mean'] = train['geo_5'].map(geo5_mean)
train['ts_demand_mean']   = train['time_slot'].map(ts_mean)
train['geo_ts_mean']      = train.set_index(['geohash', 'time_slot']).index.map(geo_ts_mean.to_dict()).values

# Apply to test (fallback to broader aggregation if unseen)
test['geo_demand_mean']  = test['geohash'].map(geo_mean).fillna(global_mean)
test['geo4_demand_mean'] = test['geo_4'].map(geo4_mean).fillna(global_mean)
test['geo5_demand_mean'] = test['geo_5'].map(geo5_mean).fillna(global_mean)
test['ts_demand_mean']   = test['time_slot'].map(ts_mean)
test['geo_ts_mean']      = test.set_index(['geohash', 'time_slot']).index.map(geo_ts_mean.to_dict())
test['geo_ts_mean']      = test['geo_ts_mean'].fillna(test['geo_demand_mean'])

print('Target encoding done.')

# ── Feature list ───────────────────────────────────────────────────────────────
FEATURES = [
    'geo_3_enc', 'geo_4_enc', 'geo_5_enc', 'geohash_enc',
    'day', 'sin_time', 'cos_time', 'time_slot', 'hour',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc',
    'Landmarks_enc', 'Temperature', 'Weather_enc',
    'geo_demand_mean', 'ts_demand_mean', 'geo4_demand_mean',
    'geo5_demand_mean', 'geo_ts_mean'
]

X      = train[FEATURES].values
y      = train['demand'].values
X_test = test[FEATURES].values

print(f'X shape: {X.shape}')
print(f'X_test shape: {X_test.shape}')

# ── Model: LightGBM ────────────────────────────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators      = 1000,
    max_depth         = 7,
    learning_rate     = 0.03,
    num_leaves        = 63,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 10,
    random_state      = 42,
    verbose           = -1
)

model.fit(X, y)

train_pred = model.predict(X)
r2    = r2_score(y, train_pred)
score = max(0, 100 * r2)
print(f'Train R²  : {r2:.4f}')
print(f'Score     : {score:.2f} / 100')

# ── Feature importance ─────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature'   : FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print('\nFeature Importances:')
print(importance_df.to_string(index=False))

# ── Generate submission ────────────────────────────────────────────────────────
preds = np.clip(model.predict(X_test), 0, 1)

submission = pd.DataFrame({
    'Index' : test['Index'],
    'demand': preds
})

submission.to_csv('submission.csv', index=False)
print('\nsubmission.csv saved!')
print(f'Shape: {submission.shape}')
print(submission.head(10))