#!/usr/bin/env python3
import os
import sys
import logging
import datetime
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
import core
importlib.reload(core)
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


    # ─── Hyper‐parameters ───────────────────────────────────────────────────────
    MODEL_NAME =  "Qwen/Qwen1.5-0.5B-Chat"#"google/gemma-2-2b"###"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    TOKENIZER_NAME = MODEL_NAME

    INPUT_SCALE = None          # “None” to trigger default calibration
    NUM_SAMPLES = 5             # number of training samples
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

    instructions = [core.get_prompt(PROMPT_TYPE, verbose=False,i=i) for i in range(NUM_SAMPLES)]
    # Load the instructions from core.get_prompt() and log the count
    logger.info(f"Loaded {len(instructions)} instructions from core.get_prompt()")
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

    # Define chat template for Gemma-2 model
    if "gemma" in MODEL_NAME.lower():
        tokenizer.chat_template = "<start_of_turn>system\n{{ messages[0]['content'] }}<end_of_turn>\n{% for message in messages[1:] %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        logger.info("Set custom chat template for Gemma model")

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

    # ─── Log some basic info about the dataset ─────────────────────────────────
    def optimize_token_for_steering_match(
        model, 
        tokenizer,
        content, 
        steering_vector, 
        source_layer_idx, 
        target_layer_idx,
        input_scale,
        max_iterations=5,
        num_candidates=200,
        similarity_metric="cosine"  # or "mse"
    ):
        """
        Finds a token that, when added to the input, produces activations similar 
        to what would be achieved by steering.
        """
        import random
        #compute the completion of unsteered activations


        # Set up SlicedModel for forward pass from source to target layer
        sliced_model = dct.SlicedModel(
            model,
            start_layer=source_layer_idx,
            end_layer=target_layer_idx,
            layers_name="model.layers"
        )
        
        # Function to compute activations for any input
        def compute_activations(text):
            model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
                x_act = hidden_states[source_layer_idx]
                y_act = hidden_states[target_layer_idx]
            return x_act, y_act
        # Optimization loop
        best_token = None
        best_similarity = float('-inf')
        best_augmented_text = None
        
        # Get vocabulary for token search
        vocab = list(tokenizer.get_vocab().keys())
        logger.info(f"Starting token optimization with {len(vocab)} candidates")
        
        for iteration in tqdm(range(max_iterations), desc="Token optimization"):
            # Sample candidate tokens
            candidate_tokens = random.sample(vocab, min(num_candidates, len(vocab)))
            
            # Try each candidate token
            for token in candidate_tokens:
                similarities = []  # Store similarities for all instructions
                
                # Test this token on all instructions[:NUM_SAMPLES]
                for inst_idx in range(len(instructions[:NUM_SAMPLES])):
                    current_content = instructions[inst_idx] if inst_idx < len(instructions) else content
                    
                    if SYSTEM_PROMPT is not None:
                        chat_init = [{'content': SYSTEM_PROMPT, 'role': 'system'}]
                    else:
                        chat_init = []
                    
                    # Build chat for current instruction
                    chat = chat_init + [{'content': current_content, 'role': 'user'}]
                    input_text = core.get_prompt_suffix(
                        tokenizer.apply_chat_template(chat,
                                                    add_special_tokens=False,
                                                    tokenize=False,
                                                    add_generation_prompt=True),
                        PROMPT_TYPE
                    )
                    
                    # Compute original activations for this instruction
                    x_orig, y_orig = compute_activations(input_text)
                    
                    # Apply steering to get steered activations (Y steered)
                    x_steered = x_orig.clone()
                    x_steered[:, -1, :] += input_scale * steering_vector
                    with torch.no_grad():
                        y_steered = sliced_model(x_steered)
                    
                    # Augment the input with the token
                    augmented_content = current_content + f" {token}"
                    chat_augmented = chat_init + [{'content': augmented_content, 'role': 'user'}]

                    augmented_text = core.get_prompt_suffix(
                        tokenizer.apply_chat_template(chat_augmented,
                                                    add_special_tokens=False,
                                                    tokenize=False,
                                                    add_generation_prompt=True),
                        PROMPT_TYPE
                    )
                    
                    # Compute new activations with added token
                    x_aug, y_aug = compute_activations(augmented_text)
                    
                    # Function to compute similarity between activations
                    def compute_similarity(a, b, metric=similarity_metric):
                        if metric == 'cosine':
                            # Focus on the final token position
                            a_token = a[:, -1, :].flatten()
                            b_token = b[:, -1, :].flatten()
                            return torch.nn.functional.cosine_similarity(a_token.unsqueeze(0), 
                                                                        b_token.unsqueeze(0))[0].item()
                        else:  # Default to MSE
                            return -torch.mean((a[:, -1, :] - b[:, -1, :]) ** 2).item()
                    
                    # Compare Y' with Y steered for this instruction
                    similarity = compute_similarity(y_aug, y_steered)
                    similarities.append(similarity)
                
                # Calculate mean similarity across all instructions
                mean_similarity = sum(similarities) / len(similarities) if similarities else float('-inf')
                
                # Update best if this mean similarity is better
                if mean_similarity > best_similarity:
                    best_similarity = mean_similarity
                    best_token = token
                    best_augmented_text = augmented_text  # Keep the last augmented text
                    logger.info(f"Iteration {iteration+1}, Found better token: '{token}' "
                              f"with mean similarity {mean_similarity:.4f}")
                    logger.info(f"Individual similarities: {[f'{s:.3f}' for s in similarities]}")
        
        # Final comparison and generation for all instructions
        if best_token:
            logger.info(f"Best token found: '{best_token}' with similarity {best_similarity:.4f}")
            
            # Generate completions to compare effects
            model_editor = dct.ModelEditor(model, layers_name="model.layers")
            
            # Prepare file for saving results - MOVE THIS INSIDE THE PROCESSING LOOP
            fn_results = os.path.join(PLOTS_DIR, "token_steering_mimicry.txt")
            
            # Open file and keep it open for all processing
            with open(fn_results, "w", encoding="utf-8") as f:
                f.write(f"Best token found: '{best_token}'\n")
                f.write(f"Similarity score: {best_similarity:.4f}\n\n")
                
                # Process each instruction
                for inst_idx in range(len(instructions[:NUM_SAMPLES])):
                    current_content = instructions[inst_idx]
                    print(current_content)
                    # Build chat for current instruction
                    if SYSTEM_PROMPT is not None:
                        chat_init = [{'content': SYSTEM_PROMPT, 'role': 'system'}]
                    else:
                        chat_init = []
                    
                    chat = chat_init + [{'content': current_content, 'role': 'user'}]
                    input_text = core.get_prompt_suffix(
                        tokenizer.apply_chat_template(chat,
                                                    add_special_tokens=False,
                                                    tokenize=False,
                                                    add_generation_prompt=True),
                        PROMPT_TYPE
                    )
                    
                    # Augmented content
                    augmented_content = current_content + f" {best_token}"
                    chat_augmented = chat_init + [{'content': augmented_content, 'role': 'user'}]
                    augmented_text = core.get_prompt_suffix(
                        tokenizer.apply_chat_template(chat_augmented,
                                                    add_special_tokens=False,
                                                    tokenize=False,
                                                    add_generation_prompt=True),
                        PROMPT_TYPE
                    )
                    
                    # Generate with steering
                    model_editor.restore()
                    model_editor.steer(input_scale * steering_vector, source_layer_idx)
                    steered_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
                    steered_output = model.generate(**steered_inputs, max_new_tokens=32, do_sample=False)
                    steered_text = tokenizer.batch_decode(steered_output, skip_special_tokens=True)[0]
                    
                    # Generate with augmented input (added token)
                    model_editor.restore()
                    augmented_inputs = tokenizer([augmented_text], return_tensors="pt").to(model.device)
                    augmented_output = model.generate(**augmented_inputs, max_new_tokens=32, do_sample=False)
                    augmented_text_completion = tokenizer.batch_decode(augmented_output, skip_special_tokens=True)[0]
                    
                    # Generate baseline (no steering or token)
                    model_editor.restore()
                    baseline_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
                    baseline_output = model.generate(**baseline_inputs, max_new_tokens=32, do_sample=False)
                    baseline_text = tokenizer.batch_decode(baseline_output, skip_special_tokens=True)[0]
                    
                    # Log results for this instruction
                    logger.info(f"Results for instruction {inst_idx+1}:")
                    logger.info(f"Baseline (no steering):\n{baseline_text}")
                    logger.info(f"Original with steering:\n{steered_text}")
                    logger.info(f"With added token (no steering):\n{augmented_text_completion}")
                    
                    # Write to file (f is still open here)
                    f.write(f"\n=== Instruction {inst_idx+1} ===\n")
                    f.write(f"Original input: {current_content}\n\n")
                    f.write("--- Baseline (no steering) ---\n")
                    f.write(baseline_text + "\n\n")
                    f.write("--- Original with steering ---\n")
                    f.write(steered_text + "\n\n")
                    f.write(f"--- With added token '{best_token}' (no steering) ---\n")
                    f.write(augmented_text_completion + "\n\n")
                    f.write("-" * 80 + "\n")
            
            logger.info(f"Results for all instructions saved to {fn_results}")
        
        return best_token, best_similarity, best_augmented_text
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
#    with torch.no_grad():
#        assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])
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


    # After fitting DCT and ranking factors
    logger.info("Finding tokens that mimic steering effects...")

    # Choose a steering vector (top-ranked factor)
    steering_vector = V[:, indices[0]]

    # Run the optimization to find the best token
    best_token, similarity, augmented_text = optimize_token_for_steering_match(
        model=model,
        tokenizer=tokenizer,
        content=instructions[0],
        steering_vector=steering_vector,
        source_layer_idx=SOURCE_LAYER_IDX,
        target_layer_idx=TARGET_LAYER_IDX,
        input_scale=INPUT_SCALE
    )

    logger.info(f"Token optimization complete: '{best_token}' with similarity {similarity:.4f}")
    """
    # You can also try with different steering vectors
    for idx in [0, 5, 10, 15]:
        logger.info(f"Testing with factor #{idx}")
        steering_vector = V[:, indices[idx]]
        token, sim, _ = optimize_token_for_steering_match(
            model=model,
            tokenizer=tokenizer,
            content=instructions[0],
            steering_vector=steering_vector,
            source_layer_idx=SOURCE_LAYER_IDX,
            target_layer_idx=TARGET_LAYER_IDX,
            input_scale=INPUT_SCALE
        )
        logger.info(f"Factor #{idx}: Best token = '{token}', similarity = {sim:.4f}")
    """

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

