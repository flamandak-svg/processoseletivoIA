import numpy as np
import tensorflow as tf

# script extra (nao faz parte dos 3 exigidos), so pra analisar
# quais digitos o modelo mais confunde entre si

model = tf.keras.models.load_model("model.h5")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

# monta a matriz na mao (linha = digito real, coluna = digito que o modelo respondeu)
matriz = np.zeros((10, 10), dtype=int)
for real, predito in zip(y_test, y_pred):
    matriz[real][predito] += 1

print("Matriz de confusao (linha = real, coluna = predito):\n")
print("     " + " ".join(f"{i:4d}" for i in range(10)))
for i in range(10):
    linha = " ".join(f"{matriz[i][j]:4d}" for j in range(10))
    print(f"{i:2d}:  {linha}")

# fora da diagonal principal = onde o modelo errou
pares = []
for i in range(10):
    for j in range(10):
        if i != j and matriz[i][j] > 0:
            pares.append((matriz[i][j], i, j))
pares.sort(reverse=True)

print("\nTop 5 confusoes mais frequentes (real -> predito : quantidade):")
for qtd, real, predito in pares[:5]:
    print(f"{real} -> {predito}: {qtd} vezes")
