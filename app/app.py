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
    
    # mapping mammal types to specific animal examples
    animal_context = {
        "Carnivore": "Bears, leopards, lions, tigers, and wolves",
        "Marsupial": "Kangaroos, koalas, opossums, wallabies, and wombats",
        "Primate": "Baboons, capuchin monkeys, chimpanzees, gorillas, and orangutans",
        "Rodent": "Beavers, mice, porcupine, rats, and squirrels",
        "Ungulate": "Deer, elk, giraffes, moose, and zebras"
    }
    
    # retrieve the list of animals for the given mammal type (if available)
    animals = animal_context.get(mammal_name, "various species")
    
    prompt = (
        f"Provide a fun fact about {mammal_name} mammals, which include animals such as {animals}. "
        "Focus on how these mammals are similar and dissimilar to humans. "
        "Respond with a simple, clear, concise, and structured response. "
        "Do not include a title such as 'Fun Fact'."
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



def generate_summary(results, actual_label, model_names):
    """
    uses Gemini to summarize the results of all models
    """
    if not HAVE_GENAI:
        return ""

    prompt = (
        "Analyze the following predictions from three models relative to "
        f"the actual label: {actual_label}.\n\n"
        "Each model output includes the predicted label, actual label, loss, "
        "and prediction probability.\n\n"
        "Please provide a short, direct summary (50 words or fewer) that includes:\n"
        "1. Which model(s) predicted correctly (if any).\n"
        "2. Any key performance differences (loss or probability).\n"
        "3. One notable insight.\n\n"
        "Model Predictions:\n"
    )

    # Instead of enumerating, use the model_names directly
    for model_name, result in zip(model_names, results):
        prompt += f"{model_name}: {result}\n"

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        summary = response.text.strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

# classify_mammal function
def classify_mammal(image, actual_label, model_names):
    """
    Always runs all three models.
    Returns:
      - A list of strings (one for each model)
      - A Gemini summary string
      - A Gemini fun fact string (if any model is correct)
    """
    text_outputs = []
    model_names = []
    fun_fact = ""

    for name, model in models.items():
        text_out, pred_label, is_correct = predict_with_model_text(
            model, 
            image, 
            actual_label
        )
        text_outputs.append(text_out)
        model_names.append(name)  # keep track of the model name in the same order

        if is_correct:
            fun_fact = generate_fun_fact(pred_label)

    # pass model_names to the updated generate_summary function
    summary = generate_summary(text_outputs, actual_label, model_names)

    return text_outputs, summary, fun_fact



def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Mammal Type Classifier")

        with gr.Row():
            # Left Column: image upload, Actual label dropdown, and Classify button.
            with gr.Column():
                gr.Markdown("""
List of animals used to train the models (rougly 50 images per animal)
- **Carnivores**: Bears, leopards, lions, tigers, and wolves 
- **Marsupials**: Kangaroos, koalas, opossums, wallabies, and wombats 
- **Primates**:  Baboons, capuchin monkeys, chimpanzees, gorillas, and orangutans 
- **Rodents**: Beavers, mice, porcupine, rats, and squirrels 
- **Ungulates**: Deer, elk, giraffes, moose, and zebras \n
(suggestion would be to google an image of an animal, take a screenshot, and paste it in below) 
        """)
                image_input = gr.Image(label="Upload a mammal image", type="pil")
                actual_label = gr.Dropdown(
                    choices=CLASS_NAMES,
                    # value="Unknown",
                    label="Actual label"
                )
                classify_button = gr.Button("Classify")

            # a row of three textboxes for model results (side by side)
            with gr.Column():
                # model results textboxes
                gr.Markdown("Model Performance")
                gr.Markdown("<div style='font-size:11px;'>P = Prediction | A = Actual | L = Loss | P = Probability</div>")
                with gr.Row():
                    result_output_1 = gr.Textbox(label="Resnet", lines=4)
                    result_output_2 = gr.Textbox(label="Efficientnet", lines=4)
                    result_output_3 = gr.Textbox(label="Mobilenet", lines=4)
                

                # gemini summary and Fun Fact textboxes
                gr.Markdown("Google Gemini LLM")
                summary_output = gr.Textbox(label="Performance Summary", lines=4)
                fun_fact_output = gr.Textbox(label="Fun Fact (if any model predicts correctly)", lines=4)

        def on_classify(img, actual):
            results, summary, fun_fact = classify_mammal(
                img, 
                actual, 
                MODEL_PATHS.keys()
                )
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
