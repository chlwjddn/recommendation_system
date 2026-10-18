"""Q2: BPR + 명시적 정규화 항 (loss = BPR loss + reg_lambda * ||emb||^2 / batch)
RecBole 기본 BPR에는 reg 항이 없어서 직접 추가함.
사용법: python run_bpr.py → bpr_results.json, q2_bpr.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


class RegBPR(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.reg_lambda = config["reg_lambda"]

    def calculate_loss(self, interaction):
        u = self.user_embedding(interaction[self.USER_ID])
        p = self.item_embedding(interaction[self.ITEM_ID])
        n = self.item_embedding(interaction[self.NEG_ITEM_ID])
        bpr = self.loss((u * p).sum(-1), (u * n).sum(-1))
        reg = (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / u.shape[0]
        return bpr + self.reg_lambda * reg


METRICS = ["recall@10", "ndcg@10", "hit@10", "precision@10"]
results = {}


def run(tag, emb, reg):
    print(f"\n===== {tag} =====")
    config = Config(model=RegBPR, dataset="ml-100k",
                    config_dict={"embedding_size": emb, "reg_lambda": reg})
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = RegBPR(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    test = trainer.evaluate(test_data, load_best_model=True)
    results[tag] = {m: float(test[m]) for m in METRICS}
    print(tag, results[tag])
    json.dump(results, open("bpr_results.json", "w"), indent=2)


embs = [32, 64, 128, 256]
regs = [0.01, 0.1, 0.5]
for e in embs:
    run(f"BPR_emb{e}", e, 0.1)
for r in regs:
    if r == 0.1:  # emb64, reg0.1은 위에서 이미 실행됨
        results[f"BPR_reg{r}"] = results["BPR_emb64"]
        continue
    run(f"BPR_reg{r}", 64, r)
json.dump(results, open("bpr_results.json", "w"), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot([str(e) for e in embs], [results[f"BPR_emb{e}"]["ndcg@10"] for e in embs], marker="o")
axes[0].set_xlabel("embedding_size (reg_lambda=0.1)"); axes[0].set_ylabel("NDCG@10")
axes[1].plot([str(r) for r in regs], [results[f"BPR_reg{r}"]["ndcg@10"] for r in regs], marker="o")
axes[1].set_xlabel("reg_lambda (embedding_size=64)"); axes[1].set_ylabel("NDCG@10")
plt.tight_layout(); plt.savefig("q2_bpr.png", dpi=150)
print("\n저장 완료: bpr_results.json, q2_bpr.png")
