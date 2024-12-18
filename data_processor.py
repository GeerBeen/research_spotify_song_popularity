import joblib
import pandas as pd


def load_models():
    """
    Завантажує моделі, дані для target encoding і стандартний scaler.
    """
    rf = joblib.load("models/main_models/random_forest_model.pkl")
    genre_pop = joblib.load("models/main_models/genre_popularity.pkl")  # Словник target encoding
    scaler_instance = joblib.load("models/main_models/scaler.pkl")
    return rf, genre_pop, scaler_instance


# Завантажуємо необхідні компоненти
rf_model, _genre_popularity, scaler = load_models()

# Ознаки, які очікує модель
REQUIRED_FEATURES = [
    'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'liveness', 'valence', 'tempo', 'track_genre_encoded',
    'tempo_duration_ratio', 'energy_danceability', 'tempo_valence',
    'loudness_squared'
]

# Значення за замовчуванням (імовірно розраховані)
DEFAULT_VALUES = {
    'duration_ms': 180000.0,
    'danceability': 0.5,
    'energy': 0.5,
    'loudness': -10.0,
    'speechiness': 0.1,
    'acousticness': 0.3,
    'liveness': 0.2,
    'valence': 0.5,
    'tempo': 120.0,
    'track_genre_encoded': 0,  # Значення за замовчуванням для жанру
    'tempo_duration_ratio': 120.0 / 180000.0,  # tempo / duration_ms
    'energy_danceability': 0.5 * 0.5,  # energy * danceability
    'tempo_valence': 120.0 * 0.5,  # tempo * valence
    'loudness_squared': (-10.0) ** 2  # loudness^2
}


class DataProcessor:
    """
    Клас для обробки і підготовки даних перед передачею в модель.
    """

    @staticmethod
    def __encode_genre(track_genre: str) -> int:
        """
        Кодує жанр за допомогою target encoding.
        Якщо жанр відсутній, повертає значення за замовчуванням.
        """
        if track_genre is None:
            return DEFAULT_VALUES['track_genre_encoded']
        encoded = _genre_popularity.get(track_genre)
        if encoded is None:
            raise ValueError(f"Невідомий жанр: {track_genre}")
        return encoded

    @staticmethod
    def preprocess_data(data: dict) -> pd.DataFrame:
        """
        Обробляє вхідні дані, заповнює відсутні значення, кодує жанр
        і обчислює додаткові ознаки.
        """
        df = pd.DataFrame([data])

        # Встановлюємо значення за замовчуванням для відсутніх ознак
        for feature, default in DEFAULT_VALUES.items():
            if feature not in df.columns or df[feature].isnull().all():
                df[feature] = default

        # Кодуємо жанр
        df['track_genre_encoded'] = DataProcessor.__encode_genre(data.get('track_genre'))
        if 'track_genre' in df.columns:
            df = df.drop(columns=['track_genre'])

        # Обчислюємо додаткові ознаки
        df['tempo_duration_ratio'] = df['tempo'] / (df['duration_ms'] + 1e-5)
        df['energy_danceability'] = df['energy'] * df['danceability']
        df['tempo_valence'] = df['tempo'] * df['valence']
        df['loudness_squared'] = df['loudness'] ** 2

        # Нормалізація числових ознак
        numerical_features = ['duration_ms', 'tempo', 'loudness']
        df[numerical_features] = scaler.transform(df[numerical_features])

        # Перевірка наявності всіх необхідних ознак
        missing_features = set(REQUIRED_FEATURES) - set(df.columns)
        if missing_features:
            raise ValueError(f"Відсутні ознаки: {missing_features}")

        df = df[REQUIRED_FEATURES]
        return df
