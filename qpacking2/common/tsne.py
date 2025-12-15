"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/12/9

# Description: 
# ------------------------------------------------------------------------------
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import os
from transformers import EsmModel, EsmTokenizer
from qpacking2.model.models import MultiTaskModel
from qpacking2.model.base import load_lora_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_multitask_model(checkpoint_dir):

    model = MultiTaskModel(
        model_dir=checkpoint_dir,
        add_lora_layers=1,
        lora_rank=4,
        lora_alpha=4,
        lora_dropout=0.1
    )

    model.backbone = load_lora_model(checkpoint_dir, add_lora_layers=1, lora_rank=4, lora_alpha=4, lora_dropout=0.1)

    # task embedding
    task_emb_path = os.path.join(checkpoint_dir, "task_embedding.pt")
    model.task_emb.load_state_dict(torch.load(task_emb_path, map_location="cpu"))

    # heads
    for task_name in model.TASKS:
        head_path = os.path.join(checkpoint_dir, "task_heads", f"{task_name}_head.pt")
        model.heads[task_name].load_state_dict(torch.load(head_path, map_location="cpu"))

    return model.to(device).eval()


def extract_task_embeddings(model, tokenizer, seqs, max_per_seq=400):
    task_embs = {t: [] for t in model.TASKS}
    task_preds = {t: [] for t in model.TASKS}

    for seq in seqs:
        inp = tokenizer(seq, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.no_grad():
            hidden = model.encode(**inp)
            B, L, H = hidden.size()
            for t_idx, task in enumerate(model.TASKS):
                head = model.heads[task]
                out = head(hidden)
                if model.TASK_TYPES[task] == "classification":
                    pred = torch.argmax(out.logits, dim=-1)
                else:
                    pred = out.logits.squeeze(-1)
                if L > max_per_seq:
                    idx = torch.randperm(L)[:max_per_seq]
                    hidden_task = hidden[0, idx, :]
                    pred_task = pred[0, idx]
                else:
                    hidden_task = hidden[0]
                    pred_task = pred[0]
                task_embs[task].append(hidden_task.cpu())
                task_preds[task].append(pred_task.cpu())

    for task in model.TASKS:
        task_embs[task] = torch.cat(task_embs[task], dim=0).numpy()
        task_preds[task] = torch.cat(task_preds[task], dim=0).numpy()
    return task_embs, task_preds


def run_tsne(X, perplex=60):
    tsne = TSNE(n_components=2, perplexity=perplex, learning_rate="auto", init="pca", random_state=200)
    return tsne.fit_transform(X)


def plot_tsne(X2d, plot_path, labels=None, values=None, title="", legend_type="pred", is_regression=False):
    plt.figure(figsize=(7, 7))
    if is_regression and values is not None:
        sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=values, cmap="plasma", s=8, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, label="Regression value")
    elif labels is not None:
        colors = ['#20B2AA', '#C71585']
        plt.scatter(X2d[:, 0], X2d[:, 1], c=[colors[int(l)] for l in labels], s=8, alpha=0.8, edgecolors='none')
        if legend_type == "pred":
            class0 = mpatches.Patch(color=colors[0], label='Surface')
            class1 = mpatches.Patch(color=colors[1], label='Interior')
            plt.legend(handles=[class1, class0])
    else:
        plt.scatter(X2d[:, 0], X2d[:, 1], s=8, alpha=0.6, edgecolors='none')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()

if __name__ == "__main__":
    official_dir  = "/Users/douzhixin/Developer/qPacking2/data/checkpoints/esm2_t30_150M_UR50D"       # 官方 ESM2
    finetuned_dir = "/Users/douzhixin/Developer/qPacking2/data/test/checkpoint-1"      # 微调后的 MultiTaskModel checkpoint
    fasta_file    = "/Users/douzhixin/Developer/qPacking-esm/data/benchmark/done/bgstrsq/bgstrsq.fasta"
    plot_dir      = "/Users/douzhixin/Developer/qPacking2/data/test/plot"
    os.makedirs(plot_dir, exist_ok=True)

    seqs = [l.strip() for l in open(fasta_file) if not l.startswith(">")]
    tokenizer = EsmTokenizer.from_pretrained(official_dir)


    print("Loading official ESM2...")
    model_official = EsmModel.from_pretrained(official_dir, add_pooling_layer=False).to(device).eval()

    print("Loading fine-tuned MultiTaskModel...")
    model_finetuned = load_multitask_model(finetuned_dir)

    for model_name, model in [("official", model_official), ("finetuned", model_finetuned)]:
        print(f"Extracting embeddings for {model_name}...")

        if model_name == "official":
            class OfficialWithHeads(torch.nn.Module):
                def __init__(self, backbone, finetuned_model):
                    super().__init__()
                    self.backbone = backbone
                    self.TASKS = finetuned_model.TASKS
                    self.TASK_TYPES = finetuned_model.TASK_TYPES
                    self.heads = finetuned_model.heads

                def encode(self, input_ids, attention_mask=None):
                    with torch.no_grad():
                        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                        hidden = hidden[:, 1:-1, :]  # remove cls eos
                    return hidden


            model_official_with_heads = OfficialWithHeads(model_official, model_finetuned).to(device).eval()
            task_embs, task_preds = extract_task_embeddings(model_official_with_heads, tokenizer, seqs)
        else:
            task_embs, task_preds = extract_task_embeddings(model, tokenizer, seqs)

        for task in model_finetuned.TASKS:
            X2d = run_tsne(task_embs[task])
            is_reg = model_finetuned.TASK_TYPES[task] == "regression"
            plot_path = os.path.join(plot_dir, f"tsne_{task}_{model_name}.png")
            plot_tsne(
                X2d,
                labels=task_preds[task] if not is_reg else None,
                values=task_preds[task] if is_reg else None,
                title=f"{task} embedding ({model_name})",
                is_regression=is_reg,
                plot_path=plot_path
            )
