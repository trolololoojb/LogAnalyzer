import csv

input_file = r'Datensätze\Cadet\ta1-cadets-e3-official.csv'
output_file = r'Datensätze\Cadet\ta1-cadets-e3-official-1000.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i < 1000:
            writer.writerow(row)
        else:
            break

print(f'Die ersten 1000 Zeilen wurden in {output_file} gespeichert.')
