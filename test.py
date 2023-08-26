from doctr.io import DocumentFile
import json
import warnings
warnings.filterwarnings("ignore")
from doctr.models import ocr_predictor
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.models.predictor import OCRPredictor

from doctr.datasets.vocabs import VOCABS

import os

os.environ["USE_TORCH"] = "1"

import torch

from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor


doc = DocumentFile.from_images('testy 5.jfif')
# Detection model
det_model = db_resnet50(pretrained=True)
#det_param = torch.load("./db_resnet50.pt", map_location="cpu")
#det_model.load_state_dict(det_param)
det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)
# detection = det_predictor(doc)


# print(detection)

#Recognition model
reco_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS['hindi'])
reco_param = torch.load("crnn_vgg16_bn_handwritten_hindi.pt", map_location="cpu")
reco_model.load_state_dict(reco_param)
reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=1, mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)), reco_model)






predictor = OCRPredictor(det_predictor, reco_predictor)

result = predictor(doc)

json_output = result.export()

result.show(doc)


#save json file

try:
    with open('output.json', 'w') as f:
        json.dump(json_output, f, indent=4)
    print("JSON output saved to output.json")
except Exception as e:
    print("An error occurred while saving the JSON output:", str(e))


