import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler

wine = datasets.load_wine()
X = wine.data
y = wine.target

mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]
X = X[:, :2]

print(f"Данные Wine (классы 0 и 1):")
print(f"Всего записей: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nОбучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка: {X_test.shape[0]} записей")

print("\n" + "=" * 70)
print("Исследование с различным количеством базовых моделей SVM")
print("=" * 70)

n_models_list = [1, 3, 5, 10, 20]
results_bagging = []

leftmost_point = min(X_test[:, 0].min(), X_train[:, 0].min())
rightmost_point = max(X_test[:, 0].max(), X_train[:, 0].max())
x_range = rightmost_point - leftmost_point
X_MIN = leftmost_point - 0.3 * x_range
X_MAX = rightmost_point + 0.3 * x_range
Y_MIN = X_train[:, 1].min() - 0.5
Y_MAX = X_train[:, 1].max() + 0.5

for idx, n_models in enumerate(n_models_list):
    if n_models == 1:
        model = SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_type = "Одиночная SVM"
    else:
        model = BaggingClassifier(
            estimator=SVC(kernel='linear', C=1.0, random_state=42),
            n_estimators=n_models,
            random_state=42,
            bootstrap=True,
            max_samples=0.8
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_type = f"Bagging ({n_models} моделей)"

    accuracy = accuracy_score(y_test, y_pred)

    results_bagging.append({
        'Количество моделей': n_models,
        'Точность': accuracy,
        'Правильно': np.sum(y_pred == y_test),
        'Неправильно': np.sum(y_pred != y_test),
        'Тип модели': model_type
    })

    print(f"\nЭксперимент {idx + 1}: {model_type}")
    print(f"  Точность: {accuracy:.4f}")
    print(f"  Правильно: {np.sum(y_pred == y_test):3d}")
    print(f"  Неправильно: {np.sum(y_pred != y_test):3d}")

    if idx == 0:
        plt.figure(figsize=(12, 8))

        xx, yy = np.meshgrid(np.linspace(X_MIN, X_MAX, 400),
                             np.linspace(Y_MIN, Y_MAX, 400))

        if n_models == 1:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm', levels=20)

            scatter = plt.scatter(X_test[:, 0], X_test[:, 1],
                                  c=y_test, cmap='coolwarm', edgecolors='k',
                                  s=120, alpha=1.0, label='Тестовая выборка')

            plt.scatter(X_train[:, 0], X_train[:, 1],
                        c=y_train, cmap='coolwarm', edgecolors='k',
                        s=40, alpha=0.3, label='Обучающая выборка')

            if hasattr(model, 'coef_'):
                w = model.coef_[0]
                b = model.intercept_[0]
                x_line = np.linspace(X_MIN, X_MAX, 200)
                y_line = (-w[0] * x_line - b) / w[1]

                plt.plot(x_line, y_line, 'k-', linewidth=3, label='Разделяющая линия')

                margin = 1.0 / np.sqrt(np.sum(w ** 2))
                plt.plot(x_line, y_line + margin, 'r--', linewidth=2, alpha=0.8,
                         label=f'Margin (±{margin:.3f})')
                plt.plot(x_line, y_line - margin, 'r--', linewidth=2, alpha=0.8)

            if hasattr(model, 'support_vectors_'):
                plt.scatter(model.support_vectors_[:, 0],
                           model.support_vectors_[:, 1],
                           s=180, facecolors='none', edgecolors='black',
                           linewidths=2.5,
                           label=f'Опорные векторы ({len(model.support_vectors_)})')

            plt.xlabel(f"{wine.feature_names[0]}", fontsize=12)
            plt.ylabel(f"{wine.feature_names[1]}", fontsize=12)

            title = f'{model_type}\nТочность: {accuracy:.4f}'

            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)

            plt.xlim([X_MIN, X_MAX])
            plt.ylim([Y_MIN, Y_MAX])

            cbar = plt.colorbar(scatter)
            cbar.set_label('Класс', fontsize=11)

            info_text = f'Моделей: {n_models}\n'
            info_text += f'Правильно: {results_bagging[-1]["Правильно"]}\n'
            info_text += f'Неправильно: {results_bagging[-1]["Неправильно"]}'

            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

plt.figure(figsize=(14, 7))

n_models = [r['Количество моделей'] for r in results_bagging]
accuracies = [r['Точность'] for r in results_bagging]

plt.plot(n_models, accuracies, 'bo-', linewidth=3, markersize=12)
plt.xlabel('Количество базовых моделей SVM', fontsize=14)
plt.ylabel('Точность классификации', fontsize=14)
plt.title('Зависимость точности от количества моделей в ансамбле',
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.xlim([0, max(n_models) + 2])
plt.xticks(n_models, fontsize=12)
plt.yticks(fontsize=12)

for i, (n, acc) in enumerate(zip(n_models, accuracies)):
    plt.text(n, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.axhline(y=accuracies[0], color='red', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Одиночная модель: {accuracies[0]:.4f}')
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ТАБЛИЦА: Исследование с различным количеством базовых моделей SVM")
print("=" * 80)
print(f"{'Эксп.':<6} {'Кол-во моделей':<15} {'Тип модели':<25} {'Точность':<12} {'Правильно':<12} {'Неправильно':<12}")
print("-" * 80)

for i, res in enumerate(results_bagging):
    print(f"{i + 1:<6} "
          f"{res['Количество моделей']:<15} "
          f"{res['Тип модели']:<25} "
          f"{res['Точность']:<12.4f} "
          f"{res['Правильно']:<12} "
          f"{res['Неправильно']:<12}")

results_stability = []
for i in range(3):
    remove_size = int(0.2 * len(X_train))
    indices = np.random.choice(len(X_train), remove_size, replace=False)

    X_train_reduced = np.delete(X_train, indices, axis=0)
    y_train_reduced = np.delete(y_train, indices)

    model_reduced = SVC(kernel='linear', C=1.0, random_state=42)
    model_reduced.fit(X_train_reduced, y_train_reduced)

    y_pred_reduced = model_reduced.predict(X_test)
    acc_reduced = accuracy_score(y_test, y_pred_reduced)

    results_stability.append({
        'Эксперимент': i + 1,
        'Обучающих записей': len(X_train_reduced),
        'Тестовых записей': len(X_test),
        'Правильно': np.sum(y_pred_reduced == y_test),
        'Неправильно': np.sum(y_pred_reduced != y_test),
        'Точность': acc_reduced
    })

print("\n" + "-" * 80)
print("ТАБЛИЦА: Проверка устойчивости")
print("-" * 80)
print(f"{'Эксп.':<6} {'Обучающих':<12} {'Тестовых':<10} {'Правильно':<12} {'Неправильно':<12} {'Точность':<10}")
print("-" * 80)

for res in results_stability:
    print(f"{res['Эксперимент']:<6} "
          f"{res['Обучающих записей']:<12} "
          f"{res['Тестовых записей']:<10} "
          f"{res['Правильно']:<12} "
          f"{res['Неправильно']:<12} "
          f"{res['Точность']:<10.4f}")
