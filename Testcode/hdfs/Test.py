
def compare_line_to_template(line_words, template_words):



    filtered_templates= [item for item in template_words if '.*\\' not in item]
    comparison_result = []
    static_counter = 0

    for lw in line_words:
        binary_labels = 0 if lw == filtered_templates[0] else 1 # Einzelne Wortübereinstimmungen in eine Liste packen
        if binary_labels == 1:
            static_counter +=1
        else:
            del filtered_templates[:1]
            static_counter = 0
        comparison_result.append(binary_labels)

        if static_counter >= 7:
            return False
    return comparison_result

print(['manhua\\.163\\.com:443\\', 'close,\\', '11971\\', 'bytes\\', '\\(11\\.6\\', 'KB\\)\\', 'sent,\\', '18220\\', 'bytes\\', '\\(17\\.7\\', 'KB\\)\\', 'received,\\', 'lifetime\\', '00:09'])
#print(['.*\\', 'close,\\', '.*\\', 'bytes\\', '.*\\', 'sent,\\', '.*\\', 'bytes\\', '.*\\', 'received,\\', 'lifetime\\', '.*', '.*', '.*'])
print(compare_line_to_template(['manhua\\.163\\.com:443\\', 'close,\\', '11971\\', 'bytes\\', '\\(11\\.6\\', 'KB\\)\\', 'sent,\\', '18220\\', 'bytes\\', '\\(17\\.7\\', 'KB\\)\\', 'received,\\', 'lifetime\\', '00:09'], ['.*\\', 'close,\\', '.*\\', 'bytes\\', '.*\\', 'sent,\\', '.*\\', 'bytes\\', '.*\\', 'received,\\', 'lifetime\\', '.*', '.*', '.*']))