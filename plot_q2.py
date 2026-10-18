"""bpr_results.json → 과제 예시 스타일의 Q2 그래프 (q2_bpr.png)"""
import json
import matplotlib.pyplot as plt

r = json.load(open("bpr_results.json"))
embs, regs = [32, 64, 128, 256], [0.01, 0.1, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
panels = [
    (axes[0], embs, [r[f"BPR_emb{e}"]["ndcg@10"] for e in embs],
     "NDCG@10 vs Embedding Size (reg_lambda=0.1)", "Embedding Size"),
    (axes[1], regs, [r[f"BPR_reg{x}"]["ndcg@10"] for x in regs],
     "NDCG@10 vs Regularization Lambda (embedding_size=64)", "Regularization Lambda"),
]
for ax, xs, ys, title, xlabel in panels:
    ax.plot(xs, ys, marker="o", color="#e0464e", label="BPR")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("NDCG@10")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("q2_bpr.png", dpi=200)
print("saved q2_bpr.png")
