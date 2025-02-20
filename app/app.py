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

# debugging 
import gradio as gr
from gradio.components import Gallery

print("Gradio version:", gr.__version__)
print("Gradio location:", gr.__file__)
print("Gallery component:", gr.Gallery)


# load env vars
from dotenv import load_dotenv
load_dotenv()

# read gemini api key from environment
genai_api_key = os.getenv("GENAI_API_KEY")

# optional: google genai
# pip install google-genai
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
    runs inference on a single model & image
    returns (fig, predicted_label, is_correct)
    """
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor()
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

    # create a figure with "model_name/pred_label/actual_label/loss/prob"
    fig = plt.figure(figsize=(4,4))
    plt.imshow(image)
    title_str = f"{model.__class__.__name__}/{pred_label}/{actual_label}/{loss_val:.2f}/{prob_val:.2f}"
    color = "black" if is_correct else "red"
    plt.title(title_str, fontsize=10, color=color)
    plt.axis("off")

    return fig, pred_label, is_correct

# gemini llm
def generate_fun_fact(mammal_name):
    """
    uses google genai to generate a fun fact, if available
    """
    if not HAVE_GENAI or not mammal_name:
        return ""
    
    prompt = (
        f"Provide a fun fact about {mammal_name} mammals related to how they "
        f"are similar and dissimilar to humans. respond with a simple, clear, "
        f"concise, and structured response."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"error generating fun fact: {e}"

# classify mammal
def classify_mammal(image, model_choice, actual_label):
    """
    main function for gradio:
    if 'compare all', predict with each model
    otherwise, use the chosen model
    returns figure(s) + optional fun_fact
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

# gradio interface
def build_interface():
    """
    returns a gradio Blocks interface
    """
    model_opts = list(MODEL_PATHS.keys()) + ["Compare All"]

    with gr.Blocks() as demo:
        gr.Markdown("# Mammal Classifier")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="upload a mammal image")
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
                outputs_gallery = gr.Gallery(
                    label="results", show_label=False
                ).scale(grid=[2], height="auto")
                fun_fact_output = gr.Textbox(
                    label="fun fact (if predicted correctly)",
                    lines=4
                )

        def on_classify(img, choice, actual):
            result, fun_fact = classify_mammal(img, choice, actual)
            
            # convert matplotlib figures to PIL images for Gradio
            if isinstance(result, list):
                # multiple
                converted = []
                for fig in result:
                    fig.canvas.draw()
                    buf = fig.canvas.tostring_rgb()
                    w, h = fig.canvas.get_width_height()
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                    from PIL import Image
                    pil_img = Image.fromarray(arr)
                    plt.close(fig)
                    converted.append(pil_img)
                return [converted, fun_fact]
            else:
                # single figure
                fig = result
                fig.canvas.draw()
                buf = fig.canvas.tostring_rgb()
                w, h = fig.canvas.get_width_height()
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                from PIL import Image
                pil_img = Image.fromarray(arr)
                plt.close(fig)
                return [pil_img, fun_fact]

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
