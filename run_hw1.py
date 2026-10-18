"""RS HW1 실험 스크립트 (RecBole, ml-100k)
사용법: python run_hw1.py  → results.json, 그래프 png 생성
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recbole.quick_start import run_recbole

METRICS = ["recall@10", "ndcg@10", "hit@10", "precision@10"]
results = {}


def run(tag, model, **params):
    print(f"\n===== {tag} =====")
    out = run_recbole(model=model, dataset="ml-100k", config_dict=params)
    results[tag] = {m: float(out["test_result"][m]) for m in METRICS}
    json.dump(results, open("results.json", "w"), indent=2)  # 중간에 죽어도 저장됨
    return results[tag]


# (1) ItemKNN / UserKNN, k 변화
ks = [1, 5, 10, 100, 500]
for k in ks:
    run(f"ItemKNN_k{k}", "ItemKNN", k=k)
    run(f"UserKNN_k{k}", "ItemKNN", k=k, method="user")

fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 3.5))
for ax, m in zip(axes, METRICS):
    for name in ["ItemKNN", "UserKNN"]:
        ax.plot(ks, [results[f"{name}_k{k}"][m] for k in ks], marker="o", label=name)
    ax.set_xscale("log"); ax.set_xlabel("k"); ax.set_title(m); ax.legend()
plt.tight_layout(); plt.savefig("q1_knn.png", dpi=150)

# (2) BPR: embedding_size, reg_lambda
# 주의: RecBole BPR 기본 설정엔 reg 항이 없을 수 있음 → bpr.py 확인 후 파라미터명 맞출 것
embs = [32, 64, 128, 256]
regs = [0.01, 0.1, 0.5]
for e in embs:
    run(f"BPR_emb{e}", "BPR", embedding_size=e, reg_lambda=0.1)
for r in regs:
    run(f"BPR_reg{r}", "BPR", embedding_size=64, reg_lambda=r)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot(embs, [results[f"BPR_emb{e}"]["ndcg@10"] for e in embs], marker="o")
axes[0].set_xlabel("embedding_size"); axes[0].set_ylabel("NDCG@10"); axes[0].set_xscale("log", base=2)
axes[1].plot([str(r) for r in regs], [results[f"BPR_reg{r}"]["ndcg@10"] for r in regs], marker="o")
axes[1].set_xlabel("reg_lambda"); axes[1].set_ylabel("NDCG@10")
plt.tight_layout(); plt.savefig("q2_bpr.png", dpi=150)

# (3) SLIM, EASE vs ItemKNN (기본 하이퍼파라미터)
run("SLIMElastic", "SLIMElastic")
run("EASE", "EASE")

print("\n===== Q3 비교 (ItemKNN은 Q1 최고 k 사용) =====")
best_k = max(ks, key=lambda k: results[f"ItemKNN_k{k}"]["ndcg@10"])
for tag in [f"ItemKNN_k{best_k}", "SLIMElastic", "EASE"]:
    print(tag, results[tag])
