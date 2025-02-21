import os
import pickle
import random
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# debugging 
from gradio.components import Gallery

print("Gradio version:", gr.__version__)
print("Gradio location:", gr.__file__)
print("Gallery component:", Gallery)

# load env vars
load_dotenv()

# read gemini api key from environment
genai_api_key = os.getenv("GENAI_API_KEY")

# optional: google genai
HAVE_GENAI = False
try:
    from google import genai
    if genai_api_key:
        client = genai.Client(api_key=genai_api_key)
        HAVE_GENAI = True
    else:
        print("warning: GENAI_API_KEY not set, fun facts won't work.")
except ImportError:
    print("warning: google-genai library not installed, fun facts won't work.")

# your class names (adjust to your dataset)
CLASS_NAMES = ['carnivore', 'marsupial', 'primate', 'rodent', 'ungulate']

# define paths to pickled models
MODEL_PATHS = {
    "resnet":        "mammal_classifier_resnet.pkl",
    "efficientnet":  "mammal_classifier_efficientnet.pkl",
    "mobilenet":     "mammal_classifier_mobilenet.pkl"
}

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # load your saved pickle files into a dict
    for name, filename in MODEL_PATHS.items():
        path = os.path.join("models", filename)  # relative to 'app.py'
        with open(path, "rb") as f:
            model = pickle.load(f)
            model.to(device)
            model.eval()
            models[name] = model

def predict_with_model(model, image, actual_label):
    """
    Runs inference on a single model & image.
    Returns (fig, predicted_label, is_correct)
    
    Updated to match training: resize, convert to tensor, then normalize
    with ImageNet stats (or your training values).
    """
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    pred_idx = output.argmax(dim=1).item()
    pred_label = CLASS_NAMES[pred_idx]
    prob_val = F.softmax(output, dim=1)[0, pred_idx].item()

    loss_val = 0.0
    is_correct = False
    if actual_label in CLASS_NAMES:
        actual_idx = CLASS_NAMES.index(actual_label)
        loss_val = F.cross_entropy(output, torch.tensor([actual_idx], device=device)).item()
        is_correct = (pred_label == actual_label)

    # Create a figure with "model_name/pred_label/actual_label/loss/prob"
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(image)
    title_str = f"{model.__class__.__name__}/{pred_label}/{actual_label}/{loss_val:.2f}/{prob_val:.2f}"
    color = "black" if is_correct else "red"
    plt.title(title_str, fontsize=10, color=color)
    plt.axis("off")

    return fig, pred_label, is_correct

# gemini llm
def generate_fun_fact(mammal_name):
    """
    Uses Google GenAI to generate a fun fact, if available.
    """
    if not HAVE_GENAI or not mammal_name:
        return ""
    
    prompt = (
        f"Provide a fun fact about {mammal_name} mammals related to how they "
        f"are similar and dissimilar to humans. Respond with a simple, clear, "
        f"concise, and structured response. Do not include a title such as 'Fun Fact'."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        fact = response.text.strip()
        
        # Remove any leading title like "Fun Fact:"
        if fact.lower().startswith("fun fact:"):
            fact = fact[len("Fun Fact:"):].strip()
        
        return fact
    except Exception as e:
        return f"Error generating fun fact: {e}"


# classify mammal
def classify_mammal(image, model_choice, actual_label):
    """
    Main function for Gradio:
    - If 'Compare All', predict with each model.
    - Otherwise, use the chosen model.
    Returns figure(s) + optional fun_fact.
    """
    results = []
    fun_fact = ""

    if model_choice == "Compare All":
        for name, model in models.items():
            fig, pred_label, is_correct = predict_with_model(model, image, actual_label)
            results.append(fig)
            if is_correct:
                fun_fact = generate_fun_fact(pred_label)
        return results, fun_fact
    else:
        model = models[model_choice]
        fig, pred_label, is_correct = predict_with_model(model, image, actual_label)
        if is_correct:
            fun_fact = generate_fun_fact(pred_label)
        return fig, fun_fact

# Gradio interface
def build_interface():
    """
    Returns a Gradio Blocks interface.
    """
    model_opts = list(MODEL_PATHS.keys()) + ["Compare All"]

    with gr.Blocks() as demo:
        gr.Markdown("# Mammal Type Classifier")

        with gr.Row():
            with gr.Column():
                # Set type="pil" so that the image is provided as a PIL image.
                image_input = gr.Image(label="upload a mammal image", type="pil")
                model_choice = gr.Dropdown(
                    choices=model_opts, 
                    value="resnet",
                    label="select model or 'compare all'"
                )
                actual_label = gr.Dropdown(
                    choices=CLASS_NAMES + ["unknown"],
                    value="unknown",
                    label="actual label (optional for loss)"
                )
                classify_button = gr.Button("classify")

            with gr.Column():
                outputs_gallery = Gallery(label="results", show_label=False)
                fun_fact_output = gr.Textbox(
                    label="fun fact (if predicted correctly)",
                    lines=4
                )

        def on_classify(img, choice, actual):
            result, fun_fact = classify_mammal(img, choice, actual)
            
            # Convert matplotlib figures to PIL images for Gradio.
            def convert_fig_to_pil(fig):
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                # Reshape to (height, width, 4) and drop the alpha channel:
                arr = buf.reshape(h, w, 4)[:, :, :3]
                pil_img = Image.fromarray(arr)
                plt.close(fig)
                return pil_img

            if isinstance(result, list):
                converted = [convert_fig_to_pil(fig) for fig in result]
                return [converted, fun_fact]
            else:
                pil_img = convert_fig_to_pil(result)
                return [[pil_img], fun_fact]

        classify_button.click(
            fn=on_classify,
            inputs=[image_input, model_choice, actual_label],
            outputs=[outputs_gallery, fun_fact_output]
        )

    return demo

if __name__ == "__main__":
    load_models()
    interface = build_interface()
    interface.launch()
