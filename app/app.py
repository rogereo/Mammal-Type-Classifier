# app.py — same design, fixed IO + robust loading (works with one or more models)

import os, pickle, gradio as gr, torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# ====== env / optional Gemini (unchanged behavior; safe if missing) ======
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")
HAVE_GENAI = False
try:
    from google import genai  # optional
    if genai_api_key:
        client = genai.Client(api_key=genai_api_key)
        HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

CLASS_NAMES = ['Carnivore', 'Marsupial', 'Primate', 'Rodent', 'Ungulate']
MODEL_PATHS = {
    "ResNet":        os.path.join("models", "mammal_classifier_resnet_100.pkl"),
    "EfficientNet":  os.path.join("models", "mammal_classifier_efficientnet_100.pkl"),
    "MobileNet":     os.path.join("models", "mammal_classifier_mobilenet_100.pkl"),
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- transforms (match minimal working demo) -----
xfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----- robust loader: pickle module OR state_dict; cpu/gpu agnostic -----
def _load_one_model(path):
    if not os.path.exists(path):
        return None, f"Model not found: {path}"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)                # full nn.Module?
        model.to(device).eval()
        return model, None
    except Exception:
        try:
            # fall back to loading a state_dict into a standard head
            from torchvision.models import resnet18
            model = resnet18(num_classes=len(CLASS_NAMES))
            state = torch.load(path, map_location="cpu")
            state = state if isinstance(state, dict) else state.get("model", state)
            model.load_state_dict(state)
            model.to(device).eval()
            return model, None
        except Exception as e2:
            return None, f"Load failed: {type(e2).__name__}: {e2}"

def load_models():
    models, errs = {}, {}
    for name, path in MODEL_PATHS.items():
        m, err = _load_one_model(path)
        models[name] = m
        if err: errs[name] = err
    return models, errs

MODELS, LOAD_ERRS = load_models()
print("[info] loaded models:", {k: bool(v) for k, v in MODELS.items()})
if LOAD_ERRS:
    for n, e in LOAD_ERRS.items():
        print(f"[warn] {n}: {e}")

# ----- core predict (accepts numpy from Gradio; convert to PIL) -----
def _predict_text(model, pil_img, actual_label):
    x = xfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx]
    prob_val = float(probs[pred_idx].item())

    loss_val = 0.0
    if actual_label in CLASS_NAMES:
        actual_idx = CLASS_NAMES.index(actual_label)
        with torch.no_grad():
            loss_val = float(F.cross_entropy(out, torch.tensor([actual_idx], device=device)).item())

    return (
        f"P  |   {pred_label}\n"
        f"A  |   {actual_label}\n"
        f"L  |   {loss_val:.2f}\n"
        f"P  |   {prob_val:.2f}"
    ), pred_label

def classify_click(img_np, actual_label):
    # convert NumPy -> PIL RGB (fix for older Gradio image types)
    pil_img = Image.fromarray(img_np).convert("RGB")

    ordered = ["ResNet", "EfficientNet", "MobileNet"]
    outputs = []
    correct_pred_label = None

    for name in ordered:
        model = MODELS.get(name)
        if model is None:
            msg = LOAD_ERRS.get(name, "Model not available")
            outputs.append(f"Error: {msg}")
            continue
        try:
            text, pred = _predict_text(model, pil_img, actual_label or "")
            outputs.append(text)
            if HAVE_GENAI and actual_label and pred == actual_label and correct_pred_label is None:
                correct_pred_label = pred
        except Exception as e:
            outputs.append(f"Error: {type(e).__name__}: {e}")

    # optional Gemini summaries/facts (no-op if not configured)
    summary = ""
    fun_fact = ""
    if HAVE_GENAI:
        try:
            prompt = (
                f"You are analyzing classification results for the actual label: {actual_label}.\n\n"
                "Each model output below has four fields:\n"
                "  P = Predicted label\n"
                "  A = Actual label\n"
                "  L = Cross-entropy loss (lower is better)\n"
                "  P = Prediction probability for the predicted label\n\n"
                "Model Outputs:\n" +
                "\n".join(f"{name}:\n{o}" for name, o in zip(ordered, outputs)) +
                "\n\nPlease write a short, clear summary (≤50 words) that includes:\n"
                "1. Which model(s) correctly matched the actual label.\n"
                "2. Key performance differences (losses and probabilities).\n"
                "3. One notable insight or caution about the predictions.\n\n"
                "Ensure your answer directly reflects the given results and is factually correct."
            )
            summary = client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text.strip()
        except Exception as e:
            summary = f"(Gemini summary error: {e})"
        if correct_pred_label:
            try:
                fact_prompt =  (
                        f"Provide a fun fact about {actual_label} mammals."
                        "Focus on how these mammals are similar and dissimilar to humans. "
                        "Respond with a simple, clear, concise, and structured response. "
                        "Do not include a title such as 'Fun Fact'."
                    )
                fun_fact = client.models.generate_content(model="gemini-2.0-flash", contents=fact_prompt).text.strip()
            except Exception as e:
                fun_fact = f"(Gemini fact error: {e})"

    # pad to 3 outputs for UI stability
    while len(outputs) < 3: outputs.append("")
    return outputs[0], outputs[1], outputs[2], summary, fun_fact

# ====== UI (same design) ======
def build_interface():
    custom_css = """
        * {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        }

        /* Main title */
        h1 {
            font-size: 32px !important;   /* Bigger title */
            font-weight: 700 !important;
            margin-bottom: 12px !important;
        }

        /* Subheadings */
        h3 {
            font-size: 20px !important;   /* Consistent size */
            font-weight: 600 !important;
            margin-top: 16px !important;
            margin-bottom: 8px !important;
        }
        """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# Mammal Type Classifier")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                    ### 🎯 Model Training Data
                    Each model was trained on ~100 images per animal:

                    **🦁 Carnivores:** Bears, Leopards, Lions, Tigers, Wolves  
                    **🦘 Marsupials:** Kangaroos, Koalas, Opossums, Wallabies, Wombats  
                    **🐵 Primates:** Baboons, Capuchin Monkeys, Chimpanzees, Gorillas, Orangutans  
                    **🐭 Rodents:** Beavers, Mice, Porcupines, Rats, Squirrels  
                    **🦌 Ungulates:** Deer, Elk, Giraffes, Moose, Zebras
                """)
                gr.Markdown("### 📋 Instructions")
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("Step 1")
                        gr.Markdown("Upload an image of a mammal from the list above")
                        # IMPORTANT: numpy type for stable image handling
                        image_input = gr.Image(label="Mammal Image", type="numpy", image_mode="RGB")
                    with gr.Column(scale=2):
                        gr.Markdown("Step 2")
                        gr.Markdown("Select the actual mammal type from the dropdown")
                        actual_label = gr.Dropdown(choices=CLASS_NAMES, label="Actual Mammal Type")
                        gr.Markdown("Step 3")
                        gr.Markdown('Click "🔍 Classify" to see results from all three models')
                        classify_button = gr.Button("🔍 Classify")

            with gr.Column():
                gr.Markdown("### 📈 Model Performance")
                gr.Markdown("<div style='font-size:11px;'>P = Prediction | A = Actual | L = Loss | P = Probability</div>")
                with gr.Row():
                    result_output_1 = gr.Textbox(label="ResNet", lines=6)
                    result_output_2 = gr.Textbox(label="EfficientNet", lines=6)
                    result_output_3 = gr.Textbox(label="MobileNet", lines=6)
                gr.Markdown("### 🤖 Gemini AI Analysis")
                summary_output = gr.Textbox(label="📈 Model Performance Summary", lines=6)
                fun_fact_output = gr.Textbox(label="🎉 Fun Fact (when prediction is correct)", lines=6)

        classify_button.click(
            fn=classify_click,
            inputs=[image_input, actual_label],
            outputs=[result_output_1, result_output_2, result_output_3, summary_output, fun_fact_output],
        )
    return demo

if __name__ == "__main__":
    build_interface().launch(debug=True)
