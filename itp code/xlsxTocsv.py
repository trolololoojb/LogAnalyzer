#xlsx Datei wird eingelesen, Freitext und Ursache werden ausgelesen und dann in jeweils eine csv Datei geschrieben
import pandas as pd

# Excel-Datei einlesen
df = pd.read_excel(r'C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\data\Wiki_2020-23.xlsx', engine='openpyxl')

# Spaltennamen ausgeben zur Überprüfung
print("Spaltennamen in der Excel-Datei:", df.columns)

# Listen initialisieren
freitext_liste = []
ursache_liste = []

# Überprüfen, ob die Spalten existieren und die Listen befüllen
if 'Freitext' in df.columns and 'Ursache' in df.columns:
    freitext_liste = df['Freitext'].tolist()
    ursache_liste = df['Ursache'].tolist()
else:
    print("Die Spalten 'Freitext' und/oder 'Ursache' existieren nicht in der Excel-Datei.")

# Freitext Liste in eine CSV-Datei schreiben
freitext_df = pd.DataFrame(freitext_liste, columns=['Freitext'])
freitext_df.to_csv('Freitext.csv', sep=';', index=False)

# Ursache Liste in eine CSV-Datei schreiben
ursache_df = pd.DataFrame(ursache_liste, columns=['Ursache'])
ursache_df.to_csv('Ursache.csv', sep=';', index=False)

print("Die Listen wurden erfolgreich in separate CSV-Dateien geschrieben.")
