import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica, pra rodar sem tela
import matplotlib.pyplot as plt
import tensorflow as tf

# script extra (nao faz parte dos 3 exigidos), so pra visualizar
# os digitos que o modelo errou no conjunto de teste

model = tf.keras.models.load_model("model.h5")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_norm = x_test.astype("float32") / 255.0
x_test_norm = x_test_norm.reshape(-1, 28, 28, 1)

y_pred = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)

indices_errados = np.where(y_pred != y_test)[0]
print(f"Total de erros no conjunto de teste: {len(indices_errados)} de {len(y_test)}")

n_mostrar = min(10, len(indices_errados))
indices_mostrar = indices_errados[:n_mostrar]

fig, eixos = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(eixos.flat, indices_mostrar):
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(f"real={y_test[idx]} predito={y_pred[idx]}")
    ax.axis("off")

for ax in eixos.flat[n_mostrar:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("misclassified_examples.png", dpi=120)
print("Imagem salva como misclassified_examples.png")
