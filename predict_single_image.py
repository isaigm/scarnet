import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse # Para leer argumentos de la terminal

def predict_single_image(image_path, model_path='resnet_4class_randaugment_final_model.pth'):
    """
    Carga el modelo final de 4 clases y predice la clase de una sola imagen.

    Args:
        image_path (str): La ruta a la imagen que se quiere clasificar.
        model_path (str): La ruta al modelo .pth guardado.

    Returns:
        tuple: Una tupla conteniendo (nombre_clase_predicha, confianza)
    """
    # Nombres de las 4 clases finales en el orden correcto
    final_class_names = ['Boxcar', 'Hypertrophic', 'Ice Pick', 'Keloid']
    
    # Transformaciones de evaluación
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Instanciar la arquitectura del modelo
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(final_class_names))
    
    # Cargar los pesos guardados
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        return f"Error: No se encontró el archivo del modelo en {model_path}", None
        
    model.eval()

    # Cargar y procesar la imagen
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return f"Error: No se encontró el archivo de imagen en {image_path}", None

    image_tensor = eval_transforms(image).unsqueeze(0)

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        predicted_class = final_class_names[top_catid[0]]
        confidence = top_prob[0].item() * 100

    return predicted_class, confidence

# --- Punto de Entrada Principal ---
if __name__ == '__main__':
    # Configurar el parser para leer argumentos de la terminal
    parser = argparse.ArgumentParser(description='Clasifica una sola imagen de cicatriz de acné.')
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen que se desea clasificar.')
    parser.add_argument('--model', type=str, default='resnet_4class_randaugment_final_model.pth', help='Ruta al archivo del modelo entrenado .pth.')
    
    args = parser.parse_args()

    predicted_class, confidence = predict_single_image(args.image, args.model)

    if predicted_class:
        print(f"\nLa imagen en '{args.image}' ha sido clasificada como:")
        print(f"Clase: {predicted_class}")
        print(f"Confianza: {confidence:.2f}%")