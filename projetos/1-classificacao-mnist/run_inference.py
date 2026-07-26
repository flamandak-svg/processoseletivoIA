import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do MNIST
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------

N_SAMPLES = 5      # minimo exigido pelo desafio
N_ANALISE = 200    # amostras extras so pra achar um caso dificil de verdade


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interpreter = tf.lite.Interpreter(model_path=os.path.join(script_dir, "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Rodando inferencia em {N_SAMPLES} amostras usando model.tflite:\n")
    for i in range(N_SAMPLES):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(pred))
        print(f"Amostra {i + 1}: predito={predicted_class} | real={int(y_test[i])}")

    # analise extra: procura, entre um conjunto maior de amostras, o caso
    # onde o modelo ficou mais "em duvida" (menor diferenca entre a 1a e a
    # 2a classe mais provavel) -- e um jeito de achar um caso realmente
    # interessante pra comentar no relatorio, em vez de so amostras faceis
    pior_margem = None
    pior_info = None
    for i in range(N_ANALISE):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        ordenado = np.argsort(pred)[::-1]
        top1, top2 = int(ordenado[0]), int(ordenado[1])
        margem = float(pred[top1] - pred[top2])
        if pior_margem is None or margem < pior_margem:
            pior_margem = margem
            pior_info = {
                "indice": i,
                "real": int(y_test[i]),
                "predito": top1,
                "segundo_lugar": top2,
                "confianca_predito": float(pred[top1]),
                "confianca_segundo": float(pred[top2]),
            }

    print(f"\nCaso mais dificil encontrado entre as primeiras {N_ANALISE} amostras:")
    print(
        f"Amostra {pior_info['indice']}: real={pior_info['real']}, "
        f"predito={pior_info['predito']} (confianca {pior_info['confianca_predito']:.2f}), "
        f"2o colocado={pior_info['segundo_lugar']} (confianca {pior_info['confianca_segundo']:.2f})"
    )


if __name__ == "__main__":
    main()
