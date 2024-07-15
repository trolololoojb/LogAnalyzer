import re

# Ursprünglicher String
text = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"

# Ersetzen aller Zahlen durch '0'
modified_text = re.sub(r'\d', '0', text)

print(modified_text)
