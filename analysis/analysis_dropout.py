"""
Análisis de la influencia del dropout en el rendimiento de redes neuronales.
Entrena múltiples redes con diferentes tasas de dropout y compara su precisión
en datos originales y con ruido.

Uso:
    python src/ex3/analisis_dropout.py
"""
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- importar red y trainer ---
sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP
from ex3.runner_parity import load_parity_dataset, evaluate_and_print

# Configurar aspecto visual de los gráficos
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
COLORS = sns.color_palette("viridis", 6)

def train_with_dropout(X, y, dropout_rate, epochs=1000):
    """Entrena una red con la tasa de dropout especificada y devuelve resultados."""
    print(f"\n{'='*20} ENTRENANDO CON DROPOUT={dropout_rate} {'='*20}")
    
    # Configuración de la red
    layer_sizes = [35, 20, 10, 1]  # Arquitectura con 2 capas ocultas
    activations = ["", "tanh", "tanh", "sigmoid"]
    
    # Inicializar modelo
    net = MLP(layer_sizes, activations, dropout_rate=dropout_rate)
    
    # Inicializar trainer
    trainer = Trainer(
        net=net,
        loss_name="mse",
        optimizer_name="adam",
        optim_kwargs={"learning_rate": 0.01},
        batch_size=10,
        max_epochs=epochs,
        log_every=100
    )
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    loss_history = trainer.fit(X, y)
    training_time = time.time() - start_time
    
    return {
        "net": net,
        "loss_history": loss_history,
        "training_time": training_time
    }

def evaluate_on_datasets(model, datasets):
    """Evalúa el modelo en múltiples conjuntos de datos y devuelve las métricas."""
    results = {}
    
    for name, (X, y) in datasets.items():
        y_hat = model.forward(X)
        y_pred = (y_hat > 0.5).astype(int)
        accuracy = np.mean(y_pred == y)
        results[name] = accuracy
    
    return results

def main():
    # Configurar rutas de datos
    base_path = Path("data")
    main_file = base_path / "TP3-ej3-digitos.txt"
    
    if not main_file.exists():
        print(f"Error: no se encuentra el archivo base en {main_file}")
        sys.exit(1)
    
    # Cargar datos originales
    print("Cargando datos originales...")
    X, y = load_parity_dataset(main_file)
    
    # Cargar datos con ruido
    noisy_datasets = {}
    for i in range(1, 5):
        noisy_file = base_path / f"noisy{i}.txt"
        if noisy_file.exists():
            X_noisy, y_noisy = load_parity_dataset(noisy_file)
            noisy_datasets[f"noisy{i}"] = (X_noisy, y_noisy)
    
    # Incluir dataset original en la lista completa
    all_datasets = {"original": (X, y), **noisy_datasets}
    
    # Definir tasas de dropout a probar
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Almacenar resultados
    results = {}
    all_loss_histories = {}
    training_times = []
    
    for rate in dropout_rates:
        # Entrenar con esta tasa de dropout
        model_results = train_with_dropout(X, y, rate)
        
        # Evaluar en todos los datasets
        accuracy_results = evaluate_on_datasets(model_results["net"], all_datasets)
        
        # Guardar resultados
        results[rate] = accuracy_results
        all_loss_histories[rate] = model_results["loss_history"]
        training_times.append(model_results["training_time"])
    
    # Visualización 1: Comparación de precisión por tasa de dropout
    plt.figure(figsize=(14, 8))
    for i, dataset in enumerate(all_datasets.keys()):
        accuracies = [results[rate][dataset] for rate in dropout_rates]
        plt.plot(dropout_rates, accuracies, marker='o', linewidth=2, label=dataset, color=COLORS[i])
    
    plt.title("Efecto del Dropout en la Precisión", fontsize=16)
    plt.xlabel("Tasa de Dropout", fontsize=14)
    plt.ylabel("Precisión", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig("dropout_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Visualización 2: Curvas de pérdida durante el entrenamiento
    plt.figure(figsize=(14, 8))
    for i, rate in enumerate(dropout_rates):
        history = all_loss_histories[rate]
        plt.plot(history, label=f"Dropout {rate}", linewidth=2, alpha=0.8, color=COLORS[i])
    
    plt.title("Curvas de Convergencia con Diferentes Tasas de Dropout", fontsize=16)
    plt.xlabel("Época", fontsize=14)
    plt.ylabel("Pérdida", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yscale('log')
    plt.savefig("dropout_convergence_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Visualización 3: Tiempo de entrenamiento
    plt.figure(figsize=(14, 6))
    plt.bar(
        [f"Dropout {rate}" for rate in dropout_rates], 
        training_times, 
        color=COLORS
    )
    plt.title("Tiempo de Entrenamiento por Tasa de Dropout", fontsize=16)
    plt.xlabel("Tasa de Dropout", fontsize=14)
    plt.ylabel("Tiempo (segundos)", fontsize=14)
    plt.grid(axis='y')
    plt.savefig("dropout_training_time.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Imprimir tabla de resultados
    print("\n\n" + "="*80)
    print("RESUMEN DE RESULTADOS".center(80))
    print("="*80)
    
    # Encabezados
    print(f"{'Dropout Rate':<12}", end="")
    for dataset in all_datasets.keys():
        print(f"{dataset.capitalize():<10}", end="")
    print(f"{'Tiempo (s)':<10}")
    
    # Datos
    for i, rate in enumerate(dropout_rates):
        print(f"{rate:<12.1f}", end="")
        for dataset in all_datasets.keys():
            print(f"{results[rate][dataset]:<10.4f}", end="")
        print(f"{training_times[i]:<10.2f}")
    
    print("="*80)
    print("\nAnálisis completado. Gráficos guardados.")

if __name__ == "__main__":
    main()