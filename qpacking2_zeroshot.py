"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/20

# Description: 
# ------------------------------------------------------------------------------
"""
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

# def load_model_and_tokenizer(model_path, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForMaskedLM.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         torch_dtype=torch.float16
#     ).to(device)
#     return tokenizer, model

def load_model_and_tokenizer(model_path, device, official):
    """
    加载 MaskedLM 模型和 tokenizer，支持官方和 LoRA 两种情况。
    LoRA 只注入最后 add_lora_layers
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if official:
        # 官方模型直接加载
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        return tokenizer, model

    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32  # LoRA 用 float32 更稳定
    )

    base_model = model.base_model
    num_layers = len(base_model.encoder.layer)
    target_modules = []

    for i in range(max(0, num_layers - 3), num_layers):
        layer_prefix = f"encoder.layer.{i}."
        target_modules.extend([
            f"{layer_prefix}attention.self.query",
            f"{layer_prefix}attention.self.key",
            f"{layer_prefix}attention.self.value",
            f"{layer_prefix}attention.output.dense",
            f"{layer_prefix}output.dense"
        ])

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )

    base_model = get_peft_model(base_model, lora_config)
    model.base_model = base_model
    model = model.to(device)

    return tokenizer, model


def get_token_probs_wt_marginals(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
        token_probs = torch.log_softmax(logits, dim=-1)

    return token_probs


def get_token_probs_masked_marginals(model, tokenizer, input_ids):
    all_token_probs = []
    for i in tqdm(range(input_ids.size(1))):
        masked_inputs = input_ids.clone()
        masked_inputs[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=masked_inputs).logits

            probs = torch.log_softmax(logits, dim=-1)
        all_token_probs.append(probs[:, i])
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    return token_probs


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx):
    wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
    assert sequence[idx] == wt, f"Wildtype {wt} does not match sequence at position {idx}: {sequence[idx]}"
    wt_encoded = tokenizer.convert_tokens_to_ids(wt)
    mt_encoded = tokenizer.convert_tokens_to_ids(mt)
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()


def score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name):
    df[model_name] = df[mutation_col].apply(
        lambda mutation: label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx)
    )
    return df


def score_with_pseudo_ppl(df, sequence, model, tokenizer, offset_idx, mutation_col, model_name):
    tqdm.pandas()
    df[model_name] = df[mutation_col].progress_apply(
        lambda mutation: compute_pppl(mutation, sequence, model, tokenizer, offset_idx)
    )
    return df






def main(model_path, official, model_name, sequence, dms_input, mutation_col, scoring_strategy, dms_output):
    df = pd.read_csv(dms_input)
    offset_idx = df.loc[0, "pos"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(model_path, device, official)
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if scoring_strategy == "wt-marginals":
        token_probs = get_token_probs_wt_marginals(model, inputs)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(df.head(5))

    elif scoring_strategy == "masked-marginals":
        token_probs = get_token_probs_masked_marginals(model, tokenizer, input_ids)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

    elif scoring_strategy == "pseudo-ppl":
        df = score_with_pseudo_ppl(df, sequence, model, tokenizer, offset_idx, mutation_col, model_name)

    logger.info(f"Writing to file: {dms_output}")
    df.to_csv(dms_output, index=False)

    valid_df = df.dropna(subset=[model_name])
    if not valid_df.empty:
        rho = valid_df["fitness"].corr(valid_df[model_name], method="spearman")
        print("Spearman rho:", rho)
    else:
        print("No valid predictions to compute Spearman correlation.")

def read_seq(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    seq = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return seq

def compute_embedding_cosine(model_path_before, model_path_after, sequence):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer, model_before = load_model_and_tokenizer(model_path_before, device, official=True)
    _, model_after = load_model_and_tokenizer(model_path_after, device, official=False)

    for name, param in model_after.named_parameters():
        if "lora" in name:
            print(name, param.abs().mean())

    model_before.eval()
    model_after.eval()

    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    with torch.no_grad():

        out_before = model_before(**inputs, output_hidden_states=True)
        out_after = model_after(**inputs, output_hidden_states=True)

        emb_before = out_before.hidden_states[-1]
        emb_after = out_after.hidden_states[-1]

        emb_before = emb_before[:, 1:-1, :]
        emb_after = emb_after[:, 1:-1, :]

        cos = F.cosine_similarity(
            emb_before,
            emb_after,
            dim=-1
        )

        mean_cos = cos.mean().item()

    print("Embedding cosine similarity:", mean_cos)

    return mean_cos

if __name__ == "__main__":
    # model_path = "/Users/douzhixin/Developer/qPacking2/data/checkpoints/esm2_t30_150M_UR50D"
    # official = True
    # model_path = r"/Users/douzhixin/Developer/qPacking2/data/checkpoints/adjust_weight_55111/checkpoint-3000"
    # official = False
    #
    # model_name = os.path.basename(model_path)
    # fasta_file = r"/Users/douzhixin/Developer/qPacking-esm/data/benchmark/done/tm/tm.fasta"
    # dms_input = "/Users/douzhixin/Developer/qPacking-esm/data/benchmark/done/tm/tm.csv"
    # sequence = read_seq(fasta_file)
    #
    # mutation_col = "mutant"
    # scoring_strategy = "wt-marginals"  # or "masked-marginals" or "pseudo-ppl"
    # dms_output = os.path.join(os.path.dirname(dms_input), os.path.basename(dms_input).split('.')[0] + f"_{model_name}_{scoring_strategy}_scores.csv")
    # main(model_path, official, model_name, sequence, dms_input, mutation_col, scoring_strategy, dms_output)

    base_model = "/Users/douzhixin/Developer/qPacking2/data/checkpoints/esm2_t30_150M_UR50D"

    finetuned_model = r"/Users/douzhixin/Developer/qPacking2/data/checkpoints/adjust_weight_55111/checkpoint-3000"

    fasta_file = r"/Users/douzhixin/Developer/qPacking-esm/data/benchmark/done/tm/tm.fasta"

    sequence = read_seq(fasta_file)

    compute_embedding_cosine(
        base_model,
        finetuned_model,
        sequence
    )
    import torch
    from safetensors.torch import load_file
    weights = load_file("/Users/douzhixin/Developer/qPacking2/data/checkpoints/adjust_weight_55111/checkpoint-3000/adapter_model.safetensors")

    for k in weights:
        print(k, weights[k].abs().mean())
