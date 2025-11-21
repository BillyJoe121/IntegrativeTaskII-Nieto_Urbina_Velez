import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from pathlib import Path
import glob

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
METRICS_DIR = PROJECT_ROOT / 'outputs' / 'metrics'
FIGURES_DIR = PROJECT_ROOT / 'outputs' / 'figures'

def load_metrics_summary():
    """Carga todos los JSON de métricas y crea un DataFrame comparativo."""
    metrics_files = list(METRICS_DIR.glob('*_metrics.json'))
    
    if not metrics_files:
        print(f"⚠ No se encontraron archivos JSON en {METRICS_DIR}")
        return None

    data = []
    for file_path in metrics_files:
        with open(file_path, 'r') as f:
            content = json.load(f)
            
        model_name = content.get('model_name', file_path.stem)
        perf = content.get('performance_metrics', {})
        
        data.append({
            'Model': model_name,
            'Accuracy': perf.get('accuracy', 0),
            'F1 Score': perf.get('f1_score', 0),
            'Precision': perf.get('precision', 0),
            'Recall': perf.get('recall', 0)
        })
    
    df = pd.DataFrame(data).set_index('Model').sort_values('Accuracy', ascending=False)
    return df

def show_figures():
    """Muestra todas las imágenes .png guardadas en outputs/figures"""
    figures = list(FIGURES_DIR.glob('*.png'))
    
    if not figures:
        print(f"⚠ No se encontraron imágenes en {FIGURES_DIR}")
        return

    print(f"\nMostrando {len(figures)} imágenes guardadas...\n")
    
    for img_path in figures:
        try:
            img = mpimg.imread(str(img_path))
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Archivo: {img_path.name}", fontsize=12)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error mostrando {img_path.name}: {e}")

def main():
    print("\n" + "="*60)
    print("  RESUMEN DE RESULTADOS DEL PROYECTO")
    print("="*60 + "\n")

    # 1. Tabla Comparativa
    df = load_metrics_summary()
    if df is not None:
        print("TABLA COMPARATIVA DE MODELOS:")
        print("-" * 60)
        print(df)
        print("-" * 60)
        
        # Gráfica rápida de comparación
        df.plot(kind='bar', y=['Accuracy', 'F1 Score'], figsize=(10, 6), rot=0)
        plt.title("Comparación de Rendimiento: Accuracy vs F1")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 2. Mostrar Imágenes Generadas
    show_figures()
    
    print("\n✅ Visualización completada.")

if __name__ == "__main__":
    main()