import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

def gini(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# Лучший признак и порог
def best_split(X, y):
    best_g = 1
    best_f = None
    best_t = None
    best_left = None
    best_right = None

    n, m = X.shape

    for f in range(m):
        values = np.unique(X[:, f])
        for t in values:
            left = X[:, f] <= t
            right = X[:, f] > t

            if sum(left) == 0 or sum(right) == 0:
                continue

            g_left = gini(y[left])
            g_right = gini(y[right])

            w_g = (sum(left) / n) * g_left + (sum(right) / n) * g_right

            if w_g < best_g:
                best_g = w_g
                best_f = f
                best_t = t
                best_left = left
                best_right = right

    return best_f, best_t, best_left, best_right, best_g


def build_tree(X, y, max_d, d=0):
    if d >= max_d or len(np.unique(y)) == 1 or len(y) < 2:
        # Лист
        cls = np.argmax(np.bincount(y))
        return {'type': 'leaf', 'class': cls, 'samples': len(y)}

    f, t, left, right, g = best_split(X, y)

    if f is None:
        cls = np.argmax(np.bincount(y))
        return {'type': 'leaf', 'class': cls, 'samples': len(y)}

    # Рекурсия
    left_tree = build_tree(X[left], y[left], max_d, d + 1)
    right_tree = build_tree(X[right], y[right], max_d, d + 1)

    return {
        'type': 'node',
        'feature': f,
        'threshold': t,
        'left': left_tree,
        'right': right_tree,
        'gini': g,
        'samples': len(y)
    }


def predict_one(tree, x):
    while tree['type'] == 'node':
        if x[tree['feature']] <= tree['threshold']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['class']


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def draw_tree(tree, x=0.5, y=0.9, width=0.4, depth=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    if tree['type'] == 'leaf':
        color = ['lightgreen', 'yellow', 'pink'][tree['class']]
        ax.text(x, y, f"Класс {tree['class']+1}\nn={tree['samples']}",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                ha='center', va='center', fontsize=10)
    else:
        ax.text(x, y, f"{features[tree['feature']]}\n<= {tree['threshold']:.2f}\n"
                      f"Gini={tree['gini']:.3f}\nn={tree['samples']}",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
                ha='center', va='center', fontsize=9)

        # Левый ребёнок
        if tree['left']:
            left_x = x - width / 2
            left_y = y - 0.15
            ax.plot([x, left_x], [y - 0.02, left_y + 0.02], 'k-', lw=1)
            ax.text((x + left_x) / 2, (y + left_y) / 2 - 0.02, "Да",
                    ha='center', va='center', fontsize=8, color='green')
            draw_tree(tree['left'], left_x, left_y, width / 2, depth + 1, ax)

        # Правый ребёнок
        if tree['right']:
            right_x = x + width / 2
            right_y = y - 0.15
            ax.plot([x, right_x], [y - 0.02, right_y + 0.02], 'k-', lw=1)
            ax.text((x + right_x) / 2, (y + right_y) / 2 - 0.02, "Нет",
                    ha='center', va='center', fontsize=8, color='red')
            draw_tree(tree['right'], right_x, right_y, width / 2, depth + 1, ax)


data = load_wine()
X = data.data
y = data.target
features = data.feature_names

# Случайное разделение на обучающую/тестовую
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

depths = [1, 2, 3, 4, 5] # Задаём количество уровней
results = []

for d in depths:
    tree = build_tree(X_train, y_train, d)

    train_pred = predict(tree, X_train)
    test_pred = predict(tree, X_test)

    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    results.append((d, train_acc, test_acc))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Дерево решений CART (глубина {d})", fontsize=14, pad=20)

    draw_tree(tree, ax=ax)
    plt.tight_layout()
    plt.show()


print("\n=== Таблица результатов ===")
print("Глубина | Обучающая | Тестовая")
print("-" * 30)
for d, tr, te in results:
    print(f"{d:7} | {tr:9.3f} | {te:8.3f}")


plt.figure(figsize=(8, 5))
depths = [r[0] for r in results]
train_acc = [r[1] for r in results]
test_acc = [r[2] for r in results]

plt.plot(depths, train_acc, 'o-', label='Обучающая', linewidth=2)
plt.plot(depths, test_acc, 's-', label='Тестовая', linewidth=2)
plt.xlabel('Глубина дерева')
plt.ylabel('Точность')
plt.title('Зависимость точности от глубины')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_depth = 3 # Вводим оптимальную глубину
print(f"\nОптимальная глубина: {best_depth}")

# Базовое дерево (вся обучающая выборка)
base_tree = build_tree(X_train, y_train, best_depth)
base_pred = predict(base_tree, X_test)
base_correct = np.sum(base_pred == y_test)
base_incorrect = len(y_test) - base_correct
base_accuracy = base_correct / len(y_test)

stability_table = []

# 3 эксперимента с удалением данных
np.random.seed(123)

for exp in range(3):
    # Удаляем 20% случайных данных из обучающей выборки
    n_remove = int(0.2 * len(X_train))
    remove_idx = np.random.choice(len(X_train), n_remove, replace=False)
    keep_idx = np.setdiff1d(np.arange(len(X_train)), remove_idx)

    X_sub = X_train[keep_idx]
    y_sub = y_train[keep_idx]

    tree_sub = build_tree(X_sub, y_sub, best_depth)
    test_pred_sub = predict(tree_sub, X_test)

    correct = np.sum(test_pred_sub == y_test)
    incorrect = len(y_test) - correct
    accuracy_sub = correct / len(y_test)

    stability_table.append({
        'Эксперимент': exp + 1,
        'Обучающая выборка': len(X_sub),
        'Тестовая выборка': len(X_test),
        'Правильно': correct,
        'Неправильно': incorrect,
        'Точность': accuracy_sub
    })

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    draw_tree(tree_sub, ax=ax)
    plt.figtext(0.5, 0.98, f"Дерево решений CART (эксперимент {exp+1})",
                ha='center', va='top', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

print("\n" + "=" * 80)
print("ТАБЛИЦА УСТОЙЧИВОСТИ ДЕРЕВА РЕШЕНИЙ")
print("=" * 80)
print(
    f"{'Эксп.':^6} | {'Обучающая':^12} | {'Тестовая':^10} | {'Правильно':^10} | {'Неправильно':^12} | {'Точность':^10}")
print("-" * 80)

print(f"{'Баз.':^6} | {len(X_train):^12} | {len(X_test):^10} | "
      f"{base_correct:^10} | {base_incorrect:^12} | {base_accuracy:^10.3f}")

for row in stability_table:
    print(f"{row['Эксперимент']:^6} | {row['Обучающая выборка']:^12} | "
          f"{row['Тестовая выборка']:^10} | {row['Правильно']:^10} | "
          f"{row['Неправильно']:^12} | {row['Точность']:^10.3f}")

print("\n" + "=" * 80)
print("АНАЛИЗ УСТОЙЧИВОСТИ")
print("=" * 80)

mean_accuracy = np.mean([row['Точность'] for row in stability_table])
mean_correct = np.mean([row['Правильно'] for row in stability_table])
mean_incorrect = np.mean([row['Неправильно'] for row in stability_table])

print(f"Базовая точность (полные данные): {base_accuracy:.3f}")
print(f"Средняя точность после удаления данных: {mean_accuracy:.3f}")
print(f"Изменение точности: {abs(base_accuracy - mean_accuracy):.3f}")

if abs(base_accuracy - mean_accuracy) < 0.05:
    print("✓ Дерево УСТОЙЧИВО: изменение точности < 5%")
else:
    print("✗ Дерево НЕУСТОЙЧИВО: изменение точности ≥ 5%")

print(f"\nСреднее правильно классифицировано: {mean_correct:.1f} из {len(y_test)}")
print(f"Среднее неправильно классифицировано: {mean_incorrect:.1f} из {len(y_test)}")
