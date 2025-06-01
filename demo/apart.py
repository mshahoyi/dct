#!/usr/bin/env python3
import os
import sys
import logging
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1) SETUP LOGGING
# ─────────────────────────────────────────────────────────────────────────────
# Create a timestamp so that each run has its own subfolder
RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Base directories for outputs
BASE_OUTPUT_DIR = os.path.join("outputs", RUN_TS)
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")

# Ensure the directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure root logger to write to both console and file
log_filename = os.path.join(LOGS_DIR, f"run_{RUN_TS}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Starting run at {RUN_TS}")
logger.info(f"Plots will be saved under: {PLOTS_DIR}")
logger.info(f"Logs will be saved under: {log_filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 2) MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ─── Imports ────────────────────────────────────────────────────────────────
    # Notebook cell markers (e.g. "# %%") are now plain comments, so they don't break.
    sys.path.append("../dct")
    sys.path.append("../src")

    import dct
    from tqdm import tqdm
    import math
    from torch import vmap
    import torch
    torch.set_default_device("cuda")
    torch.manual_seed(325)
    import pandas as pd
    import importlib

    # ─── Hyper‐parameters ───────────────────────────────────────────────────────
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    TOKENIZER_NAME = MODEL_NAME

    INPUT_SCALE = None          # “None” to trigger default calibration
    NUM_SAMPLES = 1             # number of training samples
    FORWARD_BATCH_SIZE = 1      # batch size for forward passes
    BACKWARD_BATCH_SIZE = 1     # batch size for backward passes
    MAX_SEQ_LEN = 27            # max length for truncation

    CALIBRATION_SAMPLE_SIZE = 30
    CALIBRATION_PROMPT_SAMPLE_SIZE = 1

    DIM_OUTPUT_PROJECTION = 32
    NUM_ITERS = 10
    NUM_FACTORS = 512
    FACTOR_BATCH_SIZE = 256

    SOURCE_LAYER_IDX = 10
    TARGET_LAYER_IDX = 20
    SYSTEM_PROMPT = "You are a helpful assistant"
    TOKEN_IDXS = slice(-3, None)
    NUM_EVAL = 128

    logger.info(f"MODEL_NAME = {MODEL_NAME}")
    logger.info(f"SOURCE_LAYER_IDX = {SOURCE_LAYER_IDX}, TARGET_LAYER_IDX = {TARGET_LAYER_IDX}")
    logger.info(f"NUM_FACTORS = {NUM_FACTORS}, FACTOR_BATCH_SIZE = {FACTOR_BATCH_SIZE}")

    # ─── Prepare train/test prompts ───────────────────────────────────────────────
    import core  # your module that provides `get_prompt` and `get_prompt_suffix`
    PROMPT_TYPE = 'flu'
    importlib.reload(core)

    instructions = [core.get_prompt(PROMPT_TYPE, verbose=False)]
    NUM_SAMPLES = min(NUM_SAMPLES, len(instructions))
    logger.info("Loaded instruction prompt.")

    logger.info(f"Prompt: {instructions[0]!r}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Build “chat” inputs if SYSTEM_PROMPT is used
    if SYSTEM_PROMPT is not None:
        chat_init = [{'content': SYSTEM_PROMPT, 'role': 'system'}]
    else:
        chat_init = []
    # Build examples for training
    chats = [
        chat_init + [{'content': content, 'role': 'user'}]
        for content in instructions[:NUM_SAMPLES]
    ]
    EXAMPLES = [
        core.get_prompt_suffix(
            tokenizer.apply_chat_template(chat,
                                          add_special_tokens=False,
                                          tokenize=False,
                                          add_generation_prompt=True),
            PROMPT_TYPE
        )
        for chat in chats
    ]
    # Build test examples (up to last 32)
    test_chats = [
        chat_init + [{'content': content, 'role': 'user'}]
        for content in instructions[-32:]
    ]
    TEST_EXAMPLES = [
        core.get_prompt_suffix(
            tokenizer.apply_chat_template(chat,
                                          add_special_tokens=False,
                                          tokenize=False,
                                          add_generation_prompt=True),
            PROMPT_TYPE
        )
        for chat in test_chats
    ]
    logger.info("Prepared EXAMPLES and TEST_EXAMPLES.")

    # ─── Load model ───────────────────────────────────────────────────────────────
    logger.info("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        _attn_implementation="eager"
    )
    logger.info("Model loaded.")

    # Sanity‐check hidden‐state slicing
    importlib.reload(core)
    prompt = core.get_prompt("flu", verbose=False)
    model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        hidden_states = model(model_inputs["input_ids"],
                              output_hidden_states=True).hidden_states
    sliced_model = dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
    with torch.no_grad():
        assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])
    logger.info("Sliced‐model sanity check passed.")

    # Redefine the slice we actually need
    sliced_model = dct.SlicedModel(model,
                                   start_layer=SOURCE_LAYER_IDX,
                                   end_layer=TARGET_LAYER_IDX,
                                   layers_name="model.layers")

    # ─── Build dataset of “unsteered” activations ────────────────────────────────
    d_model = model.config.hidden_size
    X = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
    Y = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")

    logger.info("Computing unsteered activations ...")
    for t in tqdm(range(0, NUM_SAMPLES, FORWARD_BATCH_SIZE)):
        with torch.no_grad():
            model_inputs = tokenizer(
                EXAMPLES[t:t + FORWARD_BATCH_SIZE],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN
            ).to(model.device)
            hidden_states = model(model_inputs["input_ids"],
                                  output_hidden_states=True).hidden_states
            h_source = hidden_states[SOURCE_LAYER_IDX]
            unsteered_target = sliced_model(h_source)

            X[t:t + FORWARD_BATCH_SIZE, :, :] = h_source.cpu()
            Y[t:t + FORWARD_BATCH_SIZE, :, :] = unsteered_target.cpu()
    logger.info("Built X and Y tensors for unsteered activations.")

    # ─── Set up ΔActivations class and vmap wrapper ──────────────────────────────
    delta_acts_single = dct.DeltaActivations(
        sliced_model,
        target_position_indices=TOKEN_IDXS
    )
    delta_acts = vmap(
        delta_acts_single,
        in_dims=(1, None, None),
        out_dims=2,
        chunk_size=FACTOR_BATCH_SIZE
    )
    logger.info("ΔActivations function set up.")

    # ─── Calibrate R if INPUT_SCALE is None ─────────────────────────────────────
    steering_calibrator = dct.SteeringCalibrator(target_ratio=.5)
    if INPUT_SCALE is None:
        logger.info("Calibrating INPUT_SCALE ...")
        INPUT_SCALE = steering_calibrator.calibrate(
            delta_acts_single,
            X.cuda(),
            Y.cuda(),
            factor_batch_size=FACTOR_BATCH_SIZE
        )
    logger.info(f"INPUT_SCALE = {INPUT_SCALE}")

    # ─── Fit Exponential DCT ─────────────────────────────────────────────────────
    logger.info("Fitting Exponential DCT ...")
    exp_dct = dct.ExponentialDCT(num_factors=NUM_FACTORS)
    U, V = exp_dct.fit(
        delta_acts_single,
        X,
        Y,
        batch_size=BACKWARD_BATCH_SIZE,
        factor_batch_size=FACTOR_BATCH_SIZE,
        init="jacobian",
        d_proj=DIM_OUTPUT_PROJECTION,
        input_scale=INPUT_SCALE,
        max_iters=NUM_ITERS,
        beta=1.0
    )
    logger.info("Exponential DCT fit complete.")

    # ─── Plot objective values ───────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    plt.plot(exp_dct.objective_values)
    plt.title("Objective values over iterations")
    fn1 = os.path.join(PLOTS_DIR, "objective_values.png")
    fig1.savefig(fn1, bbox_inches="tight")
    plt.close(fig1)
    logger.info(f"Saved plot: {fn1}")

    # ─── Visualize factor similarities (U.T @ U) ────────────────────────────────
    with torch.no_grad():
        simu = (U.t() @ U)
        mask = torch.triu(torch.ones_like(simu), diagonal=1).bool()
        simu_vals = simu[mask].cpu().numpy()

    import seaborn as sns
    fig2 = plt.figure()
    sns.histplot(simu_vals)
    plt.title("Similarity distribution for U columns (off‐diagonals)")
    fn2 = os.path.join(PLOTS_DIR, "simu_distribution.png")
    fig2.savefig(fn2, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved plot: {fn2}")

    # ─── Visualize factor similarities (V.T @ V) ────────────────────────────────
    with torch.no_grad():
        simv = (V.t() @ V)
        mask_v = torch.triu(torch.ones_like(simv), diagonal=1).bool()
        simv_vals = simv[mask_v].cpu().numpy()

    fig3 = plt.figure()
    sns.histplot(simv_vals)
    plt.title("Similarity distribution for V columns (off‐diagonals)")
    fn3 = os.path.join(PLOTS_DIR, "simv_distribution.png")
    fig3.savefig(fn3, bbox_inches="tight")
    plt.close(fig3)
    logger.info(f"Saved plot: {fn3}")

    # ─── Simple unsteered generation check ───────────────────────────────────────
    logger.info("Generating unsteered sample ...")
    model_inputs = tokenizer(EXAMPLES[:1], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info("Unsteered completion:\n%s", completion)

    # ─── Rank factors (with a custom target_vec) ─────────────────────────────────
    slice_to_end = dct.SlicedModel(
        model,
        start_layer=SOURCE_LAYER_IDX,
        end_layer=model.config.num_hidden_layers - 1,
        layers_name="model.layers"
    )
    delta_acts_end_single = dct.DeltaActivations(slice_to_end)

    SORRY_TOKEN = tokenizer.encode("Sorry", add_special_tokens=False)[0]
    SURE_TOKEN = tokenizer.encode("Sure", add_special_tokens=False)[0]
    with torch.no_grad():
        target_vec = (
            model.lm_head.weight.data[SURE_TOKEN, :]
            - model.lm_head.weight.data[SORRY_TOKEN, :]
        )

    logger.info("Ranking factors based on custom target_vec ...")
    scores, indices = exp_dct.rank(
        delta_acts_end_single,
        X,
        Y,
        target_vec=target_vec,
        batch_size=FORWARD_BATCH_SIZE,
        factor_batch_size=FACTOR_BATCH_SIZE
    )

    # ─── Plot distribution of ranking scores ────────────────────────────────────
    fig4 = plt.figure()
    sns.histplot(scores.cpu().numpy())
    plt.title("Distribution of factor‐ranking scores")
    fn4 = os.path.join(PLOTS_DIR, "ranking_scores.png")
    fig4.savefig(fn4, bbox_inches="tight")
    plt.close(fig4)
    logger.info(f"Saved plot: {fn4}")

    # ─── Model editor baseline restoration & generation ─────────────────────────
    from dct import ModelEditor
    model_editor = ModelEditor(model, layers_name="model.layers")

    d_model = model.config.hidden_size
    model_editor.restore()
    generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
    completion_act = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info("====Actual Output (baseline)====\n%s", completion_act)

    # ─── Random baseline steering ───────────────────────────────────────────────
    logger.info(f"Running random steering baseline for {NUM_EVAL} vectors ...")
    V_rand = torch.nn.functional.normalize(torch.randn(d_model, NUM_EVAL), dim=0)
    completions_rand = []
    for i in tqdm(range(NUM_EVAL), desc="Random‐baseline"):
        model_editor.restore()
        model_editor.steer(INPUT_SCALE * V_rand[:, i], SOURCE_LAYER_IDX)
        generated = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        completions_rand.append(text)
    # (Optionally save them to a text file)
    fn_rand = os.path.join(PLOTS_DIR, "random_baseline_completions.txt")
    with open(fn_rand, "w", encoding="utf-8") as outf:
        for idx, txt in enumerate(completions_rand):
            outf.write(f"====Random Vector {idx}====\n{txt}\n\n")
    logger.info(f"Saved random baseline completions to {fn_rand}")

    # ─── Steer with top factors ─────────────────────────────────────────────────
    logger.info(f"Running steering with top {NUM_EVAL} factors ...")
    completions_steered = []
    for i in tqdm(range(NUM_EVAL), desc="Steered‐baseline"):
        model_editor.restore()
        model_editor.steer(INPUT_SCALE * V[:, indices[i]], SOURCE_LAYER_IDX)
        generated = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        completions_steered.append(text)
    # Save steered completions
    fn_steered = os.path.join(PLOTS_DIR, "steered_completions.txt")
    with open(fn_steered, "w", encoding="utf-8") as outf:
        for idx, txt in enumerate(completions_steered):
            outf.write(f"====Steered by vector {idx}====\n{txt}\n\n")
    logger.info(f"Saved steered completions to {fn_steered}")

    # ─── Save one representative factor vector for future use ────────────────────
    from functools import partial
    save_vector = partial(
        core.save_vector,
        model_name=MODEL_NAME,
        source_layer_idx=SOURCE_LAYER_IDX,
        target_layer_idx=TARGET_LAYER_IDX,
        scale=INPUT_SCALE
    )
    save_vector(vector=V[:, indices[0]], concept="Medical history context")
    logger.info("Saved representative factor vector with `core.save_vector()`.")

    # ─── (Optional) Launch Gradio chat interface ────────────────────────────────
    try:
        import gradio as gr
        from transformers import pipeline

        model_editor.restore()
        chatbot = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False
        )

        def chat_fn(message, history):
            conv = [{"role": "system", "content": SYSTEM_PROMPT}]
            conv += [
                {'role': 'user' if i % 2 == 0 else 'assistant', 'content': h}
                for i, h in enumerate(history)
            ]
            conv += [{"role": "user", "content": message}]
            chat = tokenizer.apply_chat_template(
                conv,
                add_special_tokens=False,
                tokenize=False,
                add_generation_prompt=True
            )
            response = chatbot(chat, max_new_tokens=128)[0]['generated_text']
            return response

        logger.info("Launching Gradio chat interface on localhost:7860 ...")
        gr.ChatInterface(chat_fn).launch(inline=True)
    except ImportError:
        logger.warning("Gradio not installed; skipping chat interface.")

    # ─── Final evaluations: unsteered & steered detailed ────────────────────────
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Unsteered detailed
    logger.info("Running detailed unsteered eval ...")
    model_editor.restore()
    examples_all = EXAMPLES[:2] + TEST_EXAMPLES
    model_inputs = tokenizer(examples_all, return_tensors="pt", padding=True).to("cuda")
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    conts_unsteered = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    fn_un = os.path.join(PLOTS_DIR, "detailed_unsteered.txt")
    with open(fn_un, "w", encoding="utf-8") as outf:
        for cont in conts_unsteered:
            outf.write("=====Unsteered completion=====\n" + cont + "\n\n")
    logger.info(f"Saved detailed unsteered completions to {fn_un}")

    # Steered detailed (example vector indices)
    for VECIND in [0, 68, 77, 91]:
        logger.info(f"Running detailed steered eval with vector index {VECIND} ...")
        model_editor.restore()
        model_editor.steer(INPUT_SCALE * V[:, indices[VECIND]], SOURCE_LAYER_IDX)
        model_inputs = tokenizer(examples_all, return_tensors="pt", padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
        conts_steered = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        fn_s = os.path.join(PLOTS_DIR, f"detailed_steered_vec{VECIND}.txt")
        with open(fn_s, "w", encoding="utf-8") as outf:
            for cont in conts_steered:
                outf.write(f"=====Steered by vector {VECIND}=====\n{cont}\n\n")
        logger.info(f"Saved detailed steered completions to {fn_s}")

    logger.info("Run complete. All outputs (plots, logs, text dumps) are under:")
    logger.info(f"  • {BASE_OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 3) ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
