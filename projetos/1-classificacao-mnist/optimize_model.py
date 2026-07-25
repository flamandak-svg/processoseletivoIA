import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# pega o modelo que ja treinamos no passo anterior
model = tf.keras.models.load_model("model.h5")

# converte pra tflite (formato leve pra dispositivo pequeno)
# Optimize.DEFAULT sem dataset representativo aplica quantizacao
# de alcance dinamico (arredonda os pesos pra ocupar menos espaco)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado salvo como model.tflite")

# comparando os tamanhos pra colocar no relatorio depois
original_size = os.path.getsize("model.h5")
optimized_size = os.path.getsize("model.tflite")

print(f"Tamanho original  (model.h5):     {original_size / 1024:.2f} KB")
print(f"Tamanho otimizado (model.tflite): {optimized_size / 1024:.2f} KB")
print(f"Reducao: {100 * (1 - optimized_size / original_size):.1f}%")
