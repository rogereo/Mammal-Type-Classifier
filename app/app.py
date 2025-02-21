import os
import pickle
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

HAVE_GENAI = False
try:
    from google import genai
    if genai_api_key:
        client = genai.Client(api_key=genai_api_key)
        HAVE_GENAI = True
    else:
        print("warning: GENAI_API_KEY not set, fun facts and summary won't work.")
except ImportError:
    print("warning: google-genai library not installed, fun facts and summary won't work.")

CLASS_NAMES = ['Carnivore', 'Marsupial', 'Primate', 'Rodent', 'Ungulate']

MODEL_PATHS = {
    "Resnet":        "mammal_classifier_resnet.pkl",
    "Efficientnet":  "mammal_classifier_efficientnet.pkl",
    "Mobilenet":     "mammal_classifier_mobilenet.pkl"
}

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    for name, filename in MODEL_PATHS.items():
        path = os.path.join("models", filename)
        with open(path, "rb") as f:
            model = pickle.load(f)
            model.to(device)
            model.eval()
            models[name] = model


def predict_with_model_text(model, image, actual_label):
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
    prob_val = torch.softmax(output, dim=1)[0, pred_idx].item()

    loss_val = 0.0
    is_correct = False
    if actual_label in CLASS_NAMES:
        actual_idx = CLASS_NAMES.index(actual_label)
        loss_val = F.cross_entropy(output, torch.tensor([actual_idx], device=device)).item()
        is_correct = (pred_label == actual_label)

    result_text = (
        f"P  |   {pred_label}\n"
        f"A  |   {actual_label}\n"
        f"L  |   {loss_val:.2f}\n"
        f"P  |   {prob_val:.2f}"
    )
    return result_text, pred_label, is_correct


def generate_fun_fact(mammal_name):
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
        if fact.lower().startswith("fun fact:"):
            fact = fact[len("Fun Fact:"):].strip()
        return fact
    except Exception as e:
        return f"Error generating fun fact: {e}"


def generate_summary(results, actual_label):
    """
    Uses Gemini to summarize the results of all models.
    `results` is a list of text strings from each model.
    """
    if not HAVE_GENAI:
        return ""
    
    # Build a summary prompt. You can adjust this prompt as needed.
    prompt = (
        f"Given the following predictions from three models and the actual label, "
        f"please provide a concise summary comparing the predictions. "
        f"Indicate which model(s) were correct, which were incorrect, "
        f"and how they compare overall.\n\n"
        f"Actual Label: {actual_label}\n\n"
        f"Model Predictions:\n"
    )
    for idx, result in enumerate(results, start=1):
        prompt += f"Model {idx}: {result}\n"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        summary = response.text.strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"


def classify_mammal(image, actual_label):
    """
    Always runs all three models.
    Returns:
      - A list of strings (one for each model)
      - A Gemini summary string
      - A Gemini fun fact string (if any model is correct)
    """
    text_outputs = []
    fun_fact = ""
    for name, model in models.items():
        text_out, pred_label, is_correct = predict_with_model_text(model, image, actual_label)
        text_outputs.append(text_out)
        if is_correct:
            fun_fact = generate_fun_fact(pred_label)
    summary = generate_summary(text_outputs, actual_label)
    return text_outputs, summary, fun_fact


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Mammal Type Classifier")

        with gr.Row():
            # Left Column: image upload, Actual label dropdown, and Classify button.
            with gr.Column():
                image_input = gr.Image(label="Upload a mammal image", type="pil")
                actual_label = gr.Dropdown(
                    choices=CLASS_NAMES + ["Unknown"],
                    value="Unknown",
                    label="Actual label (optional for loss)"
                )
                classify_button = gr.Button("Classify")

            # Right Column:
            # 1) A row of three textboxes for model results (side by side).
            # 2) A textbox for Gemini Summary of Result.
            # 3) A textbox for Gemini Fun Fact.
            with gr.Column():
                with gr.Row():
                    result_output_1 = gr.Textbox(label="Resnet", lines=4)
                    result_output_2 = gr.Textbox(label="Efficientnet", lines=4)
                    result_output_3 = gr.Textbox(label="Mobilenet", lines=4)
                summary_output = gr.Textbox(label="Gemini Summary of Result", lines=4)
                fun_fact_output = gr.Textbox(label="Gemini Fun Fact (if predicted correctly)", lines=4)

        def on_classify(img, actual):
            results, summary, fun_fact = classify_mammal(img, actual)
            # results is always a list of 3 strings
            while len(results) < 3:
                results.append("")
            return results[0], results[1], results[2], summary, fun_fact

        classify_button.click(
            fn=on_classify,
            inputs=[image_input, actual_label],
            outputs=[result_output_1, result_output_2, result_output_3, summary_output, fun_fact_output]
        )

    return demo


if __name__ == "__main__":
    load_models()
    interface = build_interface()
    interface.launch()
