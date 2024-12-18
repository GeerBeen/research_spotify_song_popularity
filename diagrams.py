import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження даних
df = pd.read_csv('dataset.csv')

# Створення папки для збереження діаграм
import os
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)
"""
# 1. Популярність по жанрах (згруповано)

# Групування по жанрах та обчислення середньої популярності
genre_popularity = df.groupby('track_genre')['popularity'].mean().reset_index()

# Сортування за популярністю у спадаючому порядку
genre_popularity = genre_popularity.sort_values(by='popularity', ascending=False)

# Побудова стовпчастого графіка
plt.figure(figsize=(18, 8))  # Збільшимо розмір графіка
sns.barplot(data=genre_popularity, x='track_genre', y='popularity', hue='track_genre', dodge=False, palette='viridis', legend=False)
plt.xticks(rotation=90)  # Повертаємо назви жанрів вертикально
plt.title('Середня популярність по жанрах')
plt.xlabel('Жанр')
plt.ylabel('Середня популярність')
plt.tight_layout()

# Збереження графіка у файл
output_path = os.path.join(output_dir, "average_popularity_all_genres.png")
plt.savefig(output_path)
print(f"Графік збережено за шляхом: {output_path}")

# Відображення графіка
plt.show()
#  ГРАФІК ЖАНР-КІЛЬКІСТЬ ТРЕКІВ
# Групуємо дані за жанром та рахуємо кількість треків у кожному жанрі
genre_counts = df['track_genre'].value_counts().reset_index()
genre_counts.columns = ['track_genre', 'track_count']

# Будуємо графік
plt.figure(figsize=(20, 8))
sns.barplot(data=genre_counts, x='track_genre', y='track_count', palette='viridis')
plt.xticks(rotation=90)
plt.title('Кількість треків для кожного жанру')
plt.xlabel('Жанр')
plt.ylabel('Кількість треків')
plt.tight_layout()

# Збереження графіку
plt.savefig(f"{output_dir}/track_count_by_genre.png")
plt.show()
"""

# 2. Гістограма розподілу популярності
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True, color='skyblue')
plt.title('Розподіл популярності треків')
plt.xlabel('Популярність')
plt.ylabel('Кількість треків')
plt.savefig(f"{output_dir}/popularity_distribution.png")
plt.show()
"""



# 4. Теплокарта кореляції числових ознак
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy',
                      'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                      'liveness', 'valence', 'tempo']
correlation = df[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Теплокарта кореляції ознак')
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.show()

# 5. Енергія vs Танцювальність з кольором за популярністю
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=df, x='energy', y='danceability', hue='popularity', palette='viridis', alpha=0.7)
plt.title('Енергія vs Танцювальність (кольором за популярністю)')
plt.xlabel('Енергія')
plt.ylabel('Танцювальність')
plt.legend(title='Популярність')  # Додає легенду для кольорів
plt.savefig(f"{output_dir}/energy_vs_danceability.png")
plt.show()


# 6. Розподіл популярності для "explicit" (18+)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='explicit', y='popularity', ci=None)
plt.title('Середня популярність треків з/без позначки 18+')
plt.xticks([0, 1], ['Без позначки', 'Explicit'])
plt.ylabel('Середня популярність')
plt.xlabel('')
plt.savefig(f"{output_dir}/explicit_vs_popularity_simplified.png")
plt.show()



# 7. Гістограма гучності (loudness)
plt.figure(figsize=(10, 6))
sns.histplot(df['loudness'], bins=30, kde=True, color='orange')
plt.title('Розподіл гучності треків')
plt.xlabel('Гучність (дБ)')
plt.ylabel('Кількість треків')
plt.savefig(f"{output_dir}/loudness_distribution.png")
plt.show()

# 8. Додаткові діаграми (якщо потрібно)
# Можна додати діаграми для інших ознак або комбінувати їх у пару графіків."""
