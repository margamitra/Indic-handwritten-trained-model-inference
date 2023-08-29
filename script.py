from doctr.io import DocumentFile
from PIL import Image, ImageDraw
import json
import warnings

warnings.filterwarnings("ignore")
from doctr.models import ocr_predictor
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.models.predictor import OCRPredictor
from doctr.datasets.vocabs import VOCABS

import os
import torch
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor

try:
    import pytesseract
except ImportError:
    pytesseract = None


def perform_handwritten_ocr(image_path):
    doc = DocumentFile.from_images(image_path)

    # Detection model
    det_model = db_resnet50(pretrained=True)
    det_param = torch.load("./db_resnet50.pt", map_location="cpu")
    det_model.load_state_dict(det_param)
    det_predictor = DetectionPredictor(
        PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)

    # Recognition model
    reco_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS['hindi'])
    reco_param = torch.load("crnn_vgg16_bn_handwritten_hindi.pt", map_location="cpu")
    reco_model.load_state_dict(reco_param)
    reco_predictor = RecognitionPredictor(
        PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=1, mean=(0.694, 0.695, 0.693),
                     std=(0.299, 0.296, 0.301)), reco_model)

    predictor = OCRPredictor(det_predictor, reco_predictor)

    result = predictor(doc)

    json_output = result.export()

    result.show(doc)

    # Saving JSON output
    try:
        with open('output.json', 'w') as f:
            json.dump(json_output, f, indent=4)
        print("JSON output saved to output.json")
    except Exception as e:
        print("An error occurred while saving the JSON output:", str(e))


def perform_ocr_with_tesseract(image_path, lang):
    if pytesseract is not None:
        custom_config = r'--oem 3 --psm 6' #Adjust tesseract configurations according to need. I found these configs best suited for Indic OCR
        result = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT, config=custom_config)
        print("Tesseract OCR Result:")
        print(result['text'])

        # Drawing bounding boxes
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        for i in range(len(result['text'])):
            if result['text'][i].strip():
                x = result['left'][i]
                y = result['top'][i]
                w = result['width'][i]
                h = result['height'][i]
                draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        img.show()

        # Saving Tesseract output in JSON
        json_output = {'tesseract': result}
        with open('output_tesseract.json', 'w') as f:
            json.dump(json_output, f, indent=4)
    else:
        print("Pytesseract is not available. Please Install pytesseract first.")

def main():
    image_path = 'testy pr.png'  # Provide the image path

    print("Choose OCR method:")
    print("1. Doctr")
    print("2. Tesseract")
    ocr_choice = int(input("Enter your choice (1 or 2): "))

    if ocr_choice == 1:
        perform_handwritten_ocr(image_path)
    elif ocr_choice == 2:
        lang = input("Enter language code for Tesseract (default is 'eng'): ")
        perform_ocr_with_tesseract(image_path, lang)
    else:
        print("Please enter valid choice!")

if __name__ == "__main__":
    main()
