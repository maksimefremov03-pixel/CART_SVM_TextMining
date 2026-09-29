import nltk
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Три текста
text1 = """
Machine learning is transforming technology. Deep learning uses neural networks.
Reinforcement learning trains agents. Supervised learning requires labeled data.
Unsupervised learning finds patterns. Transfer learning reuses knowledge.
Learning algorithms improve with experience. Machine learning models predict.
Deep learning architectures evolve. Learning systems adapt. Machine learning advances.
Learning techniques diversify. Neural networks learn. Learning processes optimize.
Machine learning applications grow. Learning research continues.
"""

text2 = """
The little bear walked through the forest. The forest was dark and deep.
In the forest, the bear found berries. The berries were sweet and red.
The bear ate the berries. Then the bear saw a river. The river was clear and cold.
The bear drank from the river. The bear felt happy. The bear continued walking.
The bear saw a mountain. The mountain was tall and rocky. The bear climbed the mountain.
From the mountain, the bear saw the whole forest. The forest looked small from above.
The bear decided to return home. Home was warm and safe.
"""

text3 = """
Moscow is the beautiful capital of Russia. Historic Moscow has incredibly rich history.
In vibrant Moscow, visitors can explore famous Red Square. Impressive Moscow's Kremlin is world-famous.
The magnificent Moscow Metro stations are architectural masterpieces. Prestigious Moscow State University attracts students.
Moscow experiences cold winters but pleasant summers. Cultural Moscow offers diverse attractions.
Moscow architecture showcases stunning designs. Green Moscow parks provide relaxation. World-class Moscow theaters host performances.
Moscow museums exhibit priceless treasures. Busy Moscow streets buzz with activity. Dynamic Moscow life never stops.
Moscow development progresses rapidly. Tourist Moscow welcomes millions annually. Moscow never truly sleeps at night.
Moscow residents enjoy city life. Moscow events celebrate culture. Moscow bridges cross the rivers.
Moscow restaurants serve delicious food. Moscow transport connects districts. Moscow landmarks impress everyone.
Moscow weather changes frequently. Moscow history fascinates people. Moscow future looks promising.
"""

# Инициализация
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Обработка текста
def process_text(text):
    print("Исходный текст:")
    print(text[:50] + "...")

    # 1. Токенизация
    tokens = word_tokenize(text.lower())
    print(f"Токенов: {len(tokens)}")
    print(tokens)

    # 2. Удаление пунктуации и стоп-слов
    words = [w for w in tokens if w.isalpha()]
    print(f"После удаления пунктуации: {len(words)}")
    print(words)
    words = [w for w in words if w not in stop_words]
    print(f"После удаления стоп-слов: {len(words)}")
    print(words)

    # 3. Стемминг
    stemmed = [stemmer.stem(w) for w in words]
    print(f"После стемминга: {len(stemmed)}")
    print(stemmed)

    return stemmed


# Вероятности
def get_probs(words):
    n = len(words)
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    prob = {}
    for w in freq:
        prob[w] = freq[w] / n

    print("Вероятности:\n",prob)
    return prob, freq[max(freq, key=freq.get)]


# Графики
def draw_plots(probs, c, cutoff_start, cutoff_end, title):
    # Сортировка токенов
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    values = [x[1] for x in sorted_probs]
    ranks = list(range(1, len(sorted_probs) + 1))

    # Закон Ципфа
    zipf_raw = [c / r for r in ranks]
    zipf_sum = sum(zipf_raw)
    zipf_vals = [v / zipf_sum for v in zipf_raw]

    # Обычный график
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 2, 1)
    plt.plot(ranks, values, 'b-', label='Факт')
    plt.plot(ranks, zipf_vals, 'r--', label='Ципф')
    plt.axvline(x=cutoff_start, color='g')
    plt.axvline(x=cutoff_end, color='r')
    plt.title(title)
    plt.xlabel("Ранг")
    plt.ylabel("Вероятность")
    plt.legend()
    plt.grid(True)

    # log
    plt.subplot(1, 2, 2)
    plt.loglog(ranks, values, 'b-', label='Факт')
    plt.loglog(ranks, zipf_vals, 'r--', label='Ципф')
    plt.title(title + " (log)")
    plt.xlabel("log(ранг)")
    plt.ylabel("log(вероятность)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return values, zipf_vals


# MSE
def mse(actual, expected):
    total = 0
    for a, e in zip(actual, expected):
        total += (a - e) ** 2
    return total / len(actual)


all_cutoffs_start = []
all_cutoffs_end = []
all_mse = []

for i, text in enumerate([text1, text2, text3], 1):
    print(f"\nТекст {i}:")
    print("-" * 30)

    # Обработка текста
    words = process_text(text)

    # Вероятности
    probs, c = get_probs(words)

    # Уровни отсечки
    n = len(probs)
    cutoff_start = int(0.075 * n)
    cutoff_end = int(0.2 * n)
    print(f"Уровень отсечки начала: {cutoff_start}")
    print(f"Уровень отсечки конца: {cutoff_end}")

    all_cutoffs_start.append(cutoff_start)
    all_cutoffs_end.append(cutoff_end)

    # Графики
    actual, zipf_vals = draw_plots(probs, c, cutoff_start, cutoff_end, f"Текст {i}")

    # MSE
    error = mse(actual, zipf_vals)
    print(f"MSE: {error:.6f}")
    all_mse.append(error)

    # Топ слова
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Топ-5 слов:")
    for word, prob in sorted_items:
        print(f"  '{word}': {prob:.3f}")

    # Между отсечками
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print("Слова между отсечками:")
    for i, (word, prob) in enumerate(sorted_items, 1):
        if cutoff_start <= i <= cutoff_end:
            print(f"  '{word}': {prob:.3f}")

print("\n" + "=" * 50)
print("ИТОГИ")
print("=" * 50)

print(f"Средний уровень отсечки начала: {sum(all_cutoffs_start) / 3:.1f}")
print(f"Средний уровень отсечки конца: {sum(all_cutoffs_end) / 3:.1f}")
print(f"Средний MSE: {sum(all_mse) / 3:.6f}")