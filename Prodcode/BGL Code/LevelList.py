import pandas as pd

# Pfad zur CSV-Datei
csv_datei = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\Drain3 Datensätze\BGL\BGL_2k.log_structured.csv'

# Lese die CSV-Datei und spezifiziere, dass die Spalte 'Level' verwendet wird
df = pd.read_csv(csv_datei, usecols=['Level'])

# Entferne doppelte Einträge aus der Spalte 'Level'
einzigartige_levels = df['Level'].drop_duplicates()

# Pfad zur TXT-Datei, in die geschrieben werden soll
txt_datei = 'LevelList.txt'

# Schreibe die einzigartigen Level in die TXT-Datei
einzigartige_levels.to_csv(txt_datei, index=False, header=False)
