import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('dataset.csv')

df = df.drop(columns=['num', 'track_id', 'artists', 'album_name', 'track_name', 'explicit', 'mode', 'time_signature',
                      'instrumentalness', 'key'])  # попередня обробка

print("\nПропущені значення по колонкам:")
print(df.isnull().sum())

df = df.dropna()


def target_encoding(df, target_col, cat_col):  # перетворення нечислових в числові
    means = df.groupby(cat_col)[target_col].mean()
    df[cat_col + '_encoded'] = df[cat_col].map(means)
    return df


df = target_encoding(df, 'popularity', 'track_genre')

means = df.groupby('track_genre')['popularity'].mean()
genre_popularity = means.to_dict()
joblib.dump(genre_popularity, 'models/main_models/genre_popularity.pkl')

df = df.drop(columns=['track_genre'])

scaler = StandardScaler()  # масштабування даних
numerical_features = ['duration_ms', 'tempo', 'loudness']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

joblib.dump(scaler, 'models/main_models/scaler.pkl')
print("Scaler збережений.")

for col in ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness',  # обробка викидів
            'liveness', 'valence', 'tempo']:
    lower_bound = df[col].quantile(0.05)
    upper_bound = df[col].quantile(0.95)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df['tempo_duration_ratio'] = df['tempo'] / (df['duration_ms'] + 1e-5)  # створення нових ознак
df['energy_danceability'] = df['energy'] * df['danceability']
df['tempo_valence'] = df['tempo'] * df['valence']
df['loudness_squared'] = df['loudness'] ** 2

X = df.drop(columns=['popularity'])  # розділення даних
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_distributions = {  # підбір гіперпараметрів для навчання
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(  #
    RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)
print("\nПідбір гіперпараметрів...")
random_search.fit(X_train, y_train)

best_model_rf = random_search.best_estimator_
joblib.dump(best_model_rf, 'models/main_models/random_forest_model.pkl')  # збереження моделі

y_pred_rf = best_model_rf.predict(X_test)  # оцінка моделі
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"Best Parameters: {random_search.best_params_}")
print(f"Mean Squared Error (RF): {mse_rf}")
print(f"R^2 Score (RF): {r2_rf}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nВажливість ознак (Random Forest):")
print(feature_importance.to_string(index=False))
