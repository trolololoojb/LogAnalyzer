from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# Konfiguration laden
config = TemplateMinerConfig()
config.load(r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Prodcode\01_General\Drain3\drain3.ini")

# Persistenz für Templates initialisieren
persistence = FilePersistence(r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Prodcode\01_General\Drain3\drain3_state.bin")

# TemplateMiner initialisieren
template_miner = TemplateMiner(persistence, config)

# Beispielhafte Protokollmeldungen
log_messages = [
    "Error: Connection failed on port 8080",
    "Warning: Disk space low on /dev/sda1",
    "Error: Connection failed on port 8081"
]

for log_line in log_messages:
    result = template_miner.add_log_message(log_line)
    params = template_miner.extract_parameters(
    result["template_mined"], log_line, exact_matching=True)
    print(params)
    print(result)