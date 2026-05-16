import os
import argparse
import torch
import transformers
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, whoami, upload_folder, create_repo
from pathlib import Path


def login_huggingface(token=None):
    if token:
        login(token=token)
    else:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Set HF_TOKEN environment variable or pass --hf_token.")
        login(token=hf_token)


def load_base_model(model_id):
    # Load Gemma 2b-it in 4-bit with nf4 quantization to fit in GPU memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    return tokenizer, model


def test_base_model(tokenizer, model, prompt="Complete the dialogue: I'll make him an offer, he "):
    device = "cuda:0"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("Base model output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def build_lora_config():
    # Apply LoRA to all projection layers; r=8 gives a decent balance between
    # parameter count and model adaptation capacity for this task size.
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    return lora_config


def load_and_prepare_dataset(tokenizer, dataset_name="Aashi/Science_Q_and_A_dataset"):
    data = load_dataset(dataset_name)
    data = data.map(
        lambda samples: tokenizer(samples["Question"], samples["Context"]),
        batched=True,
    )
    return data


def formatting_func(example):
    # SFTTrainer expects a list of strings; format each record as an answer.
    text = f"Answer: {example['Answer'][0]}"
    return [text]


def fine_tune(model, tokenizer, data, lora_config, output_dir="outputs", max_steps=75):
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    return trainer


def run_inference(tokenizer, model, question, max_new_tokens=100):
    device = "cuda:0"
    prompt = question + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    return answer


def save_and_merge_model(model_id, trainer, save_dir="fine_tuned_science_gemma2b-it"):
    # Save LoRA adapters first, then merge into the base model for a clean checkpoint
    unmerged_dir = save_dir + "_unmerged"
    trainer.model.save_pretrained(unmerged_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    merged_model = PeftModel.from_pretrained(base_model, unmerged_dir)
    merged_model = merged_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    merged_model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    print(f"Model saved to {save_dir}")
    return save_dir


def convert_to_mediapipe(model_dir, output_dir=None, tflite_output=None):
    # MediaPipe conversion requires a separate install; import here to keep it optional.
    from mediapipe.tasks.python.genai import converter

    if output_dir is None:
        output_dir = f"/content/intermediate/{model_dir}/"
    if tflite_output is None:
        tflite_output = f"/content/{model_dir}/scigemma.bin"

    config = converter.ConversionConfig(
        input_ckpt=f"/content/{model_dir}/",
        ckpt_format="safetensors",
        model_type="GEMMA_2B",
        backend="gpu",
        output_dir=output_dir,
        combine_file_only=False,
        vocab_model_file=f"/content/{model_dir}/",
        output_tflite_file=tflite_output,
    )
    converter.convert_checkpoint(config)
    print(f"MediaPipe .bin written to {tflite_output}")


def convert_to_mlc(model_dir, quantization="q4f16_1"):
    # MLC-LLM path: convert weights → gen config → compile for Android OpenCL target.
    # Requires mlc-llm-nightly and mlc-ai-nightly to be installed separately.
    mlc_output_dir = f"{model_dir}-{quantization}-MLC"

    os.system(
        f"python -m mlc_llm convert_weight /content/{model_dir}/ "
        f"--quantization {quantization} -o /content/{mlc_output_dir}/"
    )

    os.system(
        f"mlc_llm gen_config /content/{model_dir}/ --quantization {quantization} "
        f"--conv-template gemma_instruction --context-window-size 768 "
        f"-o /content/{mlc_output_dir}/"
    )

    os.system(
        f"mlc_llm compile /content/{mlc_output_dir}/mlc-chat-config.json "
        f"--device android -o /content/{mlc_output_dir}/{model_dir}-{quantization}-android.tar"
    )
    print(f"MLC Android .tar written to /content/{mlc_output_dir}/")
    return mlc_output_dir


def push_to_huggingface(folder_path, repo_name=None, commit_message="Fine-tuned model pushed."):
    if repo_name is None:
        repo_name = os.path.basename(folder_path)

    user_info = whoami()
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"

    repo_id = create_repo(repo_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        commit_message=commit_message,
        ignore_patterns=["step_*", "epoch_*"],
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 2b-it on a science Q&A dataset and deploy to Android."
    )
    parser.add_argument("--model_id", default="google/gemma-2b-it", help="Base model ID from HF Hub.")
    parser.add_argument("--dataset", default="Aashi/Science_Q_and_A_dataset", help="Dataset name on HF Hub.")
    parser.add_argument("--output_dir", default="fine_tuned_science_gemma2b-it", help="Where to save the merged model.")
    parser.add_argument("--max_steps", type=int, default=75, help="Training steps for SFTTrainer.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token (falls back to HF_TOKEN env var).")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, load existing checkpoint.")
    parser.add_argument("--mediapipe", action="store_true", help="Convert saved model to MediaPipe .bin format.")
    parser.add_argument("--mlc", action="store_true", help="Also compile model with MLC-LLM for Android.")
    parser.add_argument("--quantization", default="q4f16_1", help="MLC quantization scheme (default: q4f16_1).")
    parser.add_argument("--push", action="store_true", help="Push fine-tuned model to Hugging Face.")
    parser.add_argument("--test_prompt", default=None, help="Run a quick inference test with this question.")
    return parser.parse_args()


def main():
    args = parse_args()

    login_huggingface(args.hf_token)

    tokenizer, model = load_base_model(args.model_id)

    # Quick sanity check on the base model before touching anything
    test_base_model(tokenizer, model)

    if not args.skip_train:
        lora_config = build_lora_config()
        data = load_and_prepare_dataset(tokenizer, args.dataset)
        trainer = fine_tune(model, tokenizer, data, lora_config, max_steps=args.max_steps)
        save_and_merge_model(args.model_id, trainer, save_dir=args.output_dir)
    else:
        print(f"Skipping training, using checkpoint at {args.output_dir}")

    if args.test_prompt:
        # Re-load from the merged checkpoint for a clean inference test
        test_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        test_model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        run_inference(test_tokenizer, test_model, args.test_prompt)

    if args.mediapipe:
        convert_to_mediapipe(model_dir=args.output_dir)

    if args.mlc:
        mlc_dir = convert_to_mlc(model_dir=args.output_dir, quantization=args.quantization)

        if args.push:
            push_to_huggingface(
                folder_path=mlc_dir,
                repo_name=f"scigemma_fine_tuned_quantized_MLC",
                commit_message="Fine-tuned quantized Gemma 2b-it model.",
            )

    if args.push and not args.mlc:
        push_to_huggingface(
            folder_path=args.output_dir,
            commit_message="Fine-tuned science Gemma 2b-it model.",
        )


if __name__ == "__main__":
    main()
