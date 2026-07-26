import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# carrega o mnist, ja vem pronto no tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# os pixels vao de 0 a 255, dividindo por 255 fica entre 0 e 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# adiciona o canal (1 = preto e branco), formato fica (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# CNN com 3 blocos: Conv2D + BatchNormalization + MaxPooling2D
# padding="same" mantem o tamanho na convolucao, so o MaxPooling
# que reduz mesmo (28 -> 14 -> 7 -> 3)
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # bloco 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # bloco 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # bloco 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),  # regularizacao antes da saida, pedido no readme
    layers.Dense(10, activation="softmax"),  # 10 classes (digitos 0-9)
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# para o treino sozinho se a perda de validacao parar de melhorar
# por 3 epocas seguidas, e volta pros pesos do melhor momento
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

# ate 15 epocas, o early stopping deve parar antes se nao tiver mais melhora
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
)

# acuracia de validacao final (pedida no relatorio)
val_accuracy_final = history.history["val_accuracy"][-1]
print(f"\nAcuracia de validacao final: {val_accuracy_final * 100:.2f}%")

# conferindo tambem no conjunto de teste (dado que o modelo nunca viu)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acuracia no teste: {test_acc * 100:.2f}%")

model.save("model.h5", save_format="h5")
print("Modelo salvo como model.h5")
