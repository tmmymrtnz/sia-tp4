import argparse
from pathlib import Path
import os

def load_digits(file_path):
    """Carga los dígitos desde un archivo, devolviendo una lista de matrices 7x5."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) % 7 != 0:
        raise ValueError(f"Formato inválido: {file_path} tiene {len(lines)} líneas, esperaba múltiplo de 7")
    
    num_digits = len(lines) // 7
    digits = []
    
    for i in range(num_digits):
        digit_matrix = []
        for j in range(7):
            row = lines[i * 7 + j].split()
            if len(row) != 5:
                raise ValueError(f"Fila {i * 7 + j + 1} tiene {len(row)} valores, esperaba 5")
            digit_matrix.append([int(bit) for bit in row])
        digits.append(digit_matrix)
    
    return digits

def generate_visualization(digits, output_file):
    """Genera un archivo con visualizaciones de dígitos en formato de 5 columnas x 2 filas."""
    # Carácter para representar píxeles encendidos
    FILLED = "█"
    EMPTY = " "
    
    # Asegurar que tenemos al menos 10 dígitos o rellenar con matrices vacías
    while len(digits) < 10:
        digits.append([[0 for _ in range(5)] for _ in range(7)])
    
    # Limitar a 10 dígitos
    digits = digits[:10]
    
    # Crear directorio de salida si no existe
    output_path = Path(output_file)
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # 2 filas de dígitos (0-4 en primera fila, 5-9 en segunda)
        for row in range(2):
            start_idx = row * 5  # 0 o 5
            
            # Para cada una de las 7 líneas que componen un dígito
            for line_idx in range(7):
                line = ""
                
                # Añadir 5 dígitos por fila, lado a lado
                for col in range(5):
                    digit_idx = start_idx + col
                    if digit_idx < len(digits):
                        line += "".join(FILLED if bit else EMPTY for bit in digits[digit_idx][line_idx])
                    
                    # Añadir espacio entre dígitos (excepto después del último)
                    if col < 4:
                        line += "   "
                
                f.write(line + "\n")
            
            # Espacio entre filas de dígitos
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Genera visualizaciones de dígitos.')
    parser.add_argument('input', help='Archivo de entrada con dígitos')
    parser.add_argument('output', help='Archivo de salida para la visualización')
    args = parser.parse_args()
    
    try:
        digits = load_digits(args.input)
        generate_visualization(digits, args.output)
        print(f"Visualización generada en {args.output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()