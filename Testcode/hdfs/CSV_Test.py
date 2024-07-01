def print_first_10_lines(file_path):
    with open(file_path, 'r') as file:
        for i in range(10):
            line = file.readline()
            if not line:  # Falls die Datei weniger als 10 Zeilen hat
                break
            print(line.strip())

# Beispielaufruf:
file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\Windows\Windows.log'
print_first_10_lines(file_path)
