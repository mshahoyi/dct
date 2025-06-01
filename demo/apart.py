# %%
import sys
sys.path.append("../src")
import dct
from tqdm import tqdm
import math
from torch import vmap
import torch
torch.set_default_device("cuda")
torch.manual_seed(325)


# %% [markdown]
# hyper-parameters

# %%
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TOKENIZER_NAME = MODEL_NAME

INPUT_SCALE = None          # norm of steering vectors; set to None to use default calibration

NUM_SAMPLES = 1             # number of training samples
FORWARD_BATCH_SIZE = 1      # batch size to use when computing forward passes
BACKWARD_BATCH_SIZE = 1     # batch size to use when computing backward passes (usually smaller)
MAX_SEQ_LEN = 27            # max length for truncating training examples

CALIBRATION_SAMPLE_SIZE = 30        # sample size for random directions used to calibrate input/output scales
CALIBRATION_PROMPT_SAMPLE_SIZE = 1  # prompt sample size for calibrating input scale

DIM_OUTPUT_PROJECTION = 32 # output projection used for approximate jacobian calculation


NUM_ITERS = 10               # number of iterations

NUM_FACTORS = 512           # number of factors to learn

SOURCE_LAYER_IDX = 10       # source layer
TARGET_LAYER_IDX = 20       # target layer


TOKEN_IDXS = slice(-3,None) # target token positions

NUM_EVAL = 64               # number of steering vectors to evaluate



# %% [markdown]
# train/test data. replace instructions with a data-set of your choice

# %%
# will use beginning/end of this dataset for train/test prompts
import pandas as pd
import sys
sys.path.append("../dct")
import importlib
import core
importlib.reload(core)

PROMPOT_TYPE = 'flu'
SYSTEM_PROMPT = core.get_sys_prompt() # system prompt; set to None for no system prompt

importlib.reload(core)
instructions = [core.get_user_prompt()]

NUM_SAMPLES = 1
print("Prompt: ", instructions[0])

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True, padding_side="left",truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

importlib.reload(core)

if SYSTEM_PROMPT is not None:
    chat_init = [{'content':SYSTEM_PROMPT, 'role':'system'}]
else:
    chat_init = []
chats = [chat_init + [{'content': content, 'role':'user'}] for content in instructions[:NUM_SAMPLES]]
EXAMPLES = [tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for chat in chats]

test_chats = [chat_init + [{'content': content, 'role':'user'}] for content in instructions[-32:]]
TEST_EXAMPLES = [tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for chat in test_chats]

print(EXAMPLES[0])

# %% [markdown]
# # load model


# %%
%%time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bnb_cfg = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             device_map="auto",
                                            #  quantization_config=bnb_cfg,
                                             trust_remote_code=True,
                                            #  torch_dtype=torch.float16,
                                             _attn_implementation="eager" # currently, `torch.func` only works well with eager attention 
                                            )


# %%
# torch.set_default_dtype(torch.float16)

# %% [markdown]
# # slice model

# %% [markdown]
# check we're correctly calculating hidden states

# %%
importlib.reload(core)
model_inputs = tokenizer(EXAMPLES[:1], return_tensors="pt").to("cuda")

with torch.no_grad():
    hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
sliced_model =dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
with torch.no_grad():
    assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])

# %%
d_model = model.config.hidden_size
generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("====Actual Output====\n", completion)

# %% [markdown]
# slice we'll actually use

# %%
sliced_model = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=TARGET_LAYER_IDX, layers_name="model.layers")

import gc
gc.collect()
torch.cuda.empty_cache()

# %%


# %% [markdown]
# # construct dataset of unsteered activations

# %%
d_model = model.config.hidden_size

X = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
Y = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
for t in tqdm(range(0, NUM_SAMPLES, FORWARD_BATCH_SIZE)):
    with torch.no_grad():
        model_inputs = tokenizer(EXAMPLES[t:t+FORWARD_BATCH_SIZE], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SEQ_LEN).to(model.device)
        hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
        h_source = hidden_states[SOURCE_LAYER_IDX] # b x t x d_model
        unsteered_target = sliced_model(h_source) # b x t x d_model

        X[t:t+FORWARD_BATCH_SIZE, :, :] = h_source
        Y[t:t+FORWARD_BATCH_SIZE, :, :] = unsteered_target



# %% [markdown]
# # class computing $\Delta_{\mathcal A}$

# %%
FACTOR_BATCH_SIZE = 32

delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=TOKEN_IDXS) # d_model, batch_size x seq_len x d_model, batch_size x seq_len x d_model
# -> batch_size x d_model
delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=FACTOR_BATCH_SIZE) # d_model x num_factors -> batch_size x d_model x num_factors

# %% [markdown]
# # calibrate $R$

# %%
steering_calibrator = dct.SteeringCalibrator(target_ratio=.5)

# %%
%%time
if INPUT_SCALE is None:
    INPUT_SCALE = steering_calibrator.calibrate(delta_acts_single,X.cuda(),Y.cuda(),factor_batch_size=FACTOR_BATCH_SIZE)


# %%
print(INPUT_SCALE)

# %% [markdown]
# # Fit Exponential DCT

# %%
%%time
exp_dct= dct.ExponentialDCT(num_factors=NUM_FACTORS)
U,V = exp_dct.fit(delta_acts_single, X, Y, batch_size=BACKWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE,
            init="jacobian", d_proj=DIM_OUTPUT_PROJECTION, input_scale=INPUT_SCALE, max_iters=10, beta=1.0)

# %% [markdown]
# Currently, `exp_dct.objective_values` only give the *causal* part of the objective (as calculating the similarity penalty would require re-calculating the $\alpha_\ell$'s in each step, which is expensive).

# %%
from matplotlib import pyplot as plt
plt.plot(exp_dct.objective_values)

gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# # Visualize factor similarities

# %%
with torch.no_grad():
    simu = (U.t() @ U)
    simu = simu[torch.triu(torch.ones_like(simu), diagonal=1).bool()]
import seaborn as sns
sns.displot(simu.cpu())


# %%
with torch.no_grad():
    simv = (V.t() @ V)
    simv = simv[torch.triu(torch.ones_like(simv), diagonal=1).bool()]
import seaborn as sns
sns.displot(simv.cpu())


# %% [markdown]
# # Evaluate

# %% [markdown]
# ## Unsteered

# %%
model_inputs = tokenizer(EXAMPLES[:1], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
completion = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)[0]
print(completion)


# %% [markdown]
# ## Rank factors

# %% [markdown]
# In general, calling `exp_dct.rank()` with `target_vec=None` ranks factors according to the $\alpha_\ell$'s. This is a reasonable choice, but a lot of the top factors will be a bit noisy.
# 
# In many applications, there will be a better proxy for interesting factors. Below, I call `exp_dct.rank()` with `target_vec` equal to the difference of the tokens "Sorry" and "Sure" (projected into residual stream space), and with `delta_acts` given by the original transformer sliced from `SOURCE_LAYER_IDX` to the *final* layer (rather than the original target layer).
# 
# This is a good proxy for whether the model will refuse.

# %%
slice_to_end = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=model.config.num_hidden_layers-1, 
                               layers_name="model.layers")
delta_acts_end_single = dct.DeltaActivations(slice_to_end) 

# %%
SORRY_TOKEN = tokenizer.encode("Sorry", add_special_tokens=False)[0]
SURE_TOKEN = tokenizer.encode("Sure", add_special_tokens=False)[0]
with torch.no_grad():
    target_vec = model.lm_head.weight.data[SURE_TOKEN,:] - model.lm_head.weight.data[SORRY_TOKEN,:]

# %%
scores, indices = exp_dct.rank(delta_acts_end_single, X, Y, target_vec=target_vec, 
                               batch_size=FORWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE)

# %%
import seaborn as sns
sns.displot(scores.cpu())

# %% [markdown]
# # Evaluate Model Edits

# %%
model_editor = dct.ModelEditor(model, layers_name="model.layers")

# %% [markdown]
# # Actual Output


# %%


# %% [markdown]
# ## Random baseline

# %%
from torch import nn

# %%
NUM_EVAL = 128
MAX_NEW_TOKENS = 32

# %%
V_rand = torch.nn.functional.normalize(torch.randn(d_model,NUM_EVAL),dim=0)
completions = []
prompt = EXAMPLES[0]
for i in tqdm(range(NUM_EVAL)):
    model_editor.restore()
    model_editor.steer(INPUT_SCALE*V_rand[:,i], SOURCE_LAYER_IDX)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, 
                           skip_special_tokens=True)[0]
    completions.append(completion)

# %%
for i in range(NUM_EVAL):
    print("====Random Vector %d, Positive :=========\n" % i)
    print(completions[i])


# %% [markdown]
# ## Steer

# %%
MAX_NEW_TOKENS = 128
from torch import nn

# %%
V.shape, V[:, indices[0]].shape
# %%
model_editor.restore()
completions = []
prompt = EXAMPLES[0]
for i in tqdm(range(256, 256+NUM_EVAL)):
    model_editor.restore()
    model_editor.steer(INPUT_SCALE*V[:,indices[i]], SOURCE_LAYER_IDX)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, 
                           skip_special_tokens=True)[0]
    completions.append(completion)

# %%
for i in range(NUM_EVAL):
    print("====Steered by vector %d=========\n" % i)
    print(completions[i][len(instructions[0]):])

#%%
from functools import partial

importlib.reload(core)
save_vector = partial(core.save_vector, model_name=MODEL_NAME, source_layer_idx=SOURCE_LAYER_IDX, target_layer_idx=TARGET_LAYER_IDX, scale=INPUT_SCALE)

save_vector(vector=V[:,indices[53]], concept="Where followed by clause 1")
save_vector(vector=V[:,indices[73]], concept="Where followed by clause 2")
save_vector(vector=V[:,indices[124]], concept="Where followed by clause 3")
save_vector(vector=V[:,indices[128+17]], concept="Where followed by clause 4")
save_vector(vector=V[:,indices[256+45]], concept="Where followed by clause 6")
save_vector(vector=V[:,indices[256+47]], concept="Where followed by clause 7")
save_vector(vector=V[:,indices[256+113]], concept="Where followed by clause 8")

V.save("V.pt")
indices.save("indices.pt")
INPUT_SCALE.save("INPUT_SCALE.pt")

# %%
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[14]], SOURCE_LAYER_IDX)

# %%
import gradio as gr
from transformers import pipeline

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False)

def chat_fn(message, history):
    conv = [{"role": "system", "content": SYSTEM_PROMPT}]
    conv += [{"role": "user", "content": message}]
    chat = tokenizer.apply_chat_template(conv, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
    # conv = [{"role": "system", "content": SYSTEM_PROMPT}]
    # conv += [{"role": "user" if i%2 == 0 else "assistant", "content": h} for i, h in enumerate(history)]
    # conv += [{"role": "user", "content": message}]
    # chat = tokenizer.apply_chat_template(conv, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
    response = chatbot(chat, max_new_tokens=128)[0]['generated_text']
    print(response)
    return response

gr.ChatInterface(chat_fn).launch(inline=True)

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## Detailed eval: unsteered

# %%
model_editor.restore()
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print("=====Unsteered completion======")
    print(cont)


# %% [markdown]
# ## Detailed eval: steering specific vectors

# %% [markdown]
# "helpful-only assistant" jailbreak

# %%
VECIND = 0
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# %% [markdown]
# "are you sure you can handle it?"

# %%
VECIND = 68
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# %% [markdown]
# list harmful instructions

# %%
VECIND = 77
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# %% [markdown]
# has propensity for bringing up nuclear weapons (but not always)

# %%
VECIND = 91
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# %%



