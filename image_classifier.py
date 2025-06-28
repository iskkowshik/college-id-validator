import base64
import io
from PIL import Image
import torch
from torchvision import transforms, models
import pytesseract
import re

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class_names = ['fake', 'genuine', 'non-id'] 


def load_model(path='id_resnet_model.pth'):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        print("PyTorch version doesn't support weights_only. Loading normally...")
        state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(" Model loaded successfully with weights")
    return model


model = load_model('id_resnet_model.pth')


def correct_orientation(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
        print(f"Image rotated by {rotation} degrees")
    except Exception as e:
        print(f"Rotation detection failed: {e}")
    return image


def predict_from_base64(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print("Image decoded and loaded")

        
        image = correct_orientation(image)

       
        img_tensor = transform(image).unsqueeze(0).to(device)

        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        
        probs_dict = {class_names[i]: round(probabilities[i].item(), 4) for i in range(len(class_names))}
        predicted_label = class_names[torch.argmax(probabilities).item()]
        genuine_confidence = round(probabilities[class_names.index('genuine')].item(), 4)

        
        print("ALL CLASS PROBABILITIES:")
        for cls, prob in probs_dict.items():
            print(f"   {cls}: {prob:.4f}")

        return {
            "final_label": predicted_label,
            "genuine_confidence": genuine_confidence,
            "all_probabilities": probs_dict
        }

    except Exception as e:
        print("Error in predict_from_base64:", e)
        raise e
