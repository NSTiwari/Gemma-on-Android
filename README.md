# Gemma on Android
This project fine-tunes the Gemma 2b-it model on a custom science Q&A dataset and deploys the fine-tuned model on Android using MediaPipe's LlmInference API.

## Pipeline:
<img src="https://github.com/NSTiwari/Gemma-on-Android/blob/main/SciGemma_Pipeline.gif"/>

## Demo Output:
<img src="https://github.com/NSTiwari/Gemma-on-Android/blob/main/SciGemma.gif" width="300" height="600"/>


## What this project does

The project has three distinct stages that work together end to end:

**1. Dataset creation (`unstructured_to_structured_dataset.py`)**

Raw science textbook PDFs are parsed with `pypdf`, split into 500-character overlapping chunks, and fed through a BERT QA model (`bert-large-uncased-whole-word-masking-finetuned-squad`) to extract question-answer pairs automatically. The model generates a "question" by decoding the highest-confidence token span from BERT's start/end logits, then answers it against the same chunk. Output is saved as both CSV and JSONL for flexibility in downstream training pipelines.

**2. Fine-tuning (`fine_tune_gemma.py`)**

Gemma 2b-it is loaded in 4-bit NF4 quantization using `BitsAndBytesConfig` to fit within a single GPU (tested on a T4). LoRA adapters (rank=8) are attached to all projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Training uses `SFTTrainer` from TRL with `paged_adamw_8bit` optimizer, 75 steps, and a batch size of 16 effective examples (4 per device × 4 gradient accumulation). After training, LoRA weights are merged back into the base model and saved as a full float16 checkpoint.

The merged checkpoint is then converted to MediaPipe's `.bin` format using `mediapipe.tasks.python.genai.converter`, which packages the model in a format that MediaPipe's `LlmInference` API can load directly on-device. There's also an optional MLC-LLM path (`--mlc`) that compiles the model to an Android OpenCL target via `mlc_llm compile`.

**3. Android app**

The app is built with Jetpack Compose and uses MediaPipe's `LlmInference` to load the `.bin` model from `/data/local/tmp/llm/` on the device. The MVVM architecture separates chat state (`ChatUiState`) from generation logic (`ChatViewModel` → `InferenceModel`). The `InferenceModel` wrapper handles async token streaming through a callback and exposes a `generateResponseAsync()` coroutine. The Compose UI shows a scrollable chat history with a persistent input bar at the bottom.


## How to run

### Prerequisites
- Python 3.10+, CUDA GPU (for fine-tuning)
- Android Studio + physical Android device with enough RAM (3GB+ free)
- Hugging Face account and token

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Create the dataset from a PDF
```bash
python unstructured_to_structured_dataset.py \
  --pdf /path/to/science_textbook.pdf \
  --csv_out qa_dataset.csv \
  --jsonl_out qa_dataset.jsonl \
  --show_sample
```

If you already have a CSV, skip PDF extraction:
```bash
python unstructured_to_structured_dataset.py \
  --from_csv Science_data.csv \
  --jsonl_out science_dataset.jsonl
```

### Step 3: Fine-tune and convert
```bash
# Fine-tune, convert to MediaPipe .bin, and push to HF Hub
python fine_tune_gemma.py \
  --model_id google/gemma-2b-it \
  --max_steps 75 \
  --mediapipe \
  --push \
  --hf_token YOUR_TOKEN
```

For MLC-LLM Android compilation (optional):
```bash
python fine_tune_gemma.py \
  --model_id google/gemma-2b-it \
  --mlc \
  --quantization q4f16_1 \
  --push
```

Test inference on the fine-tuned model:
```bash
python fine_tune_gemma.py \
  --skip_train \
  --test_prompt "What is Hemoglobin?"
```

### Step 4: Deploy on Android
1. Clone the repository and open in Android Studio.
2. Copy `scigemma.bin` to the device:
   ```bash
   adb shell mkdir -p /data/local/tmp/llm/
   adb push scigemma.bin /data/local/tmp/llm/scigemma.bin
   ```
3. Edit `InferenceModel.kt` line 44: replace `YOUR_MODE_NAME.bin` with `scigemma.bin`.
4. Build and run the app on your device.


## Project structure

```
Gemma-on-Android/
├── fine_tune_gemma.py                  # Fine-tuning + MediaPipe/MLC conversion
├── unstructured_to_structured_dataset.py  # PDF → Q&A CSV/JSONL pipeline
├── requirements.txt
└── Android_App/
    └── app/src/main/java/
        └── com/google/mediapipe/examples/llminference/
            ├── InferenceModel.kt       # MediaPipe LlmInference wrapper
            ├── ChatViewModel.kt        # Generation logic + state management
            ├── ChatScreen.kt           # Compose UI
            ├── ChatUiState.kt          # UI state data classes
            ├── LoadingScreen.kt        # Loading screen while model initializes
            └── MainActivity.kt
```


## Resources

1. Follow along three blog series explaining the code in detail:
   
   Part 1: [Step-by-Step Dataset Creation- Unstructured to Structured](https://aashi-dutt3.medium.com/part-1-step-by-step-dataset-creation-unstructured-to-structured-70abdc98abf0)

   Part 2: [Fine Tune - Gemma 2b-it model](https://aashi-dutt3.medium.com/part-2-fine-tune-gemma-2b-it-model-a26246c530e7)

   Part 3: [Deploying SciGemma on Android](https://tiwarinitin1999.medium.com/part-3-deploy-gemma-on-android-5bac532c54b7)

3. Fine-tuned model on 🤗: https://huggingface.co/NSTiwari/fine_tuned_science_gemma2b-it

4. Try our model on HFSpaces: [https://huggingface.co/spaces/Aashi/NSTiwari-fine_tuned_science_gemma2b-it?logs=container](https://huggingface.co/spaces/Aashi/NSTiwari-fine_tuned_science_gemma2b-it)

5. Checkout demo video on YouTube: https://www.youtube.com/watch?v=T_HDsVHTrwg


# Acknowledgment:
<img src="https://github.com/NSTiwari/Gemma-on-Android/blob/main/google.png">

This project was developed during Google's ML Developer Programs Gemma sprint. We thank the MLDP team for the opportunity.
