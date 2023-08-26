import json

# Load the JSON data
with open('output.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# Create HOCR content
hocr_content = []
for page in json_data['pages']:
    hocr_content.append('<div class="ocr_page">')
    for block in page['blocks']:
        block_geometry = block['geometry']
        hocr_content.append(f'<div class="ocr_carea" title="bbox {block_geometry[0][0]} {block_geometry[0][1]} {block_geometry[1][0]} {block_geometry[1][1]}">')
        for line in block['lines']:
            line_geometry = line['geometry']
            hocr_content.append(f'<div class="ocr_line" title="bbox {line_geometry[0][0]} {line_geometry[0][1]} {line_geometry[1][0]} {line_geometry[1][1]}">')
            for word in line['words']:
                word_geometry = word['geometry']
                word_bbox = f'bbox {word_geometry[0][0]} {word_geometry[0][1]} {word_geometry[1][0]} {word_geometry[1][1]}'
                word_text = f'x_wconf {word["confidence"]:.2f} {word["value"]}'
                word_hocr = f'<span class="ocrx_word" title="{word_bbox}">{word_text}</span>'
                hocr_content.append(word_hocr)
            hocr_content.append('</div>')
        hocr_content.append('</div>')
    hocr_content.append('</div>')

# Combine content and create the complete HOCR document
complete_hocr = f'<!DOCTYPE html><html><head><title></title></head><body>{" ".join(hocr_content)}</body></html>'

# Save HOCR content to a file
with open('output.html', 'w', encoding='utf-8') as hocr_file:
    hocr_file.write(complete_hocr)
