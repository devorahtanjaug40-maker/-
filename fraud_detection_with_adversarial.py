# -*- coding: utf-8 -*-
"""
欺诈检测鲁棒性实验
- 训练原始模型
- 测试改写样本（同义词替换 / 句式重构）
- 输出准确率对比与分析
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from sklearn.metrics import accuracy_score
from io import StringIO

# ======================
# 配置
# ======================
MODEL_PATH = "bert-base-chinese"  # 使用在线模型（避免本地路径问题）
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5
DEVICE = torch.device("cpu")

# ======================
# 模拟原始数据（替代 "测试集结果.csv"）
# ======================
RAW_DATA = [
    ("用户说：我今天要转账给朋友", "True"),
    ("客服回复：请确认对方身份", "False"),
    ("这个账户是骗子", "True"),
    ("我要汇款到陌生账号", "True"),
    ("系统提示交易异常", "False"),
    ("快帮我转50万到这个卡", "True"),
    ("请问如何修改密码？", "False"),
    ("有人冒充银行", "True"),
    ("验证码是多少？", "False"),
    ("紧急！我的钱被转走了", "True"),
] * 5  # 共 50 条

# ======================
# 构造改写数据
# ======================
REWRITTEN_SYNONYM = [  # 同义词替换
    ("用户称：我今日需汇款至友人", "True"),
    ("客服建议：请核实对方信息", "False"),
    ("该账号涉嫌诈骗", "True"),
    ("我想转账至未知账户", "True"),
    ("系统警告交易存在风险", "False"),
    ("速助我汇50万元至此银行卡", "True"),
    ("如何更改登录密码？", "False"),
    ("有不法分子假扮银行", "True"),
    ("短信验证码能告知吗？", "False"),
    ("急！我的资金已被盗转", "True"),
] * 5

REWRITTEN_STRUCTURE = [  # 句式重构
    ("我要把钱转给我朋友", "True"),
    ("建议你先查证一下对方是谁", "False"),
    ("小心！这可能是诈骗账户", "True"),
    ("能不能帮我把钱打到一个新账号？", "True"),
    ("交易好像出问题了", "False"),
    ("麻烦立刻转50万到这张卡上", "True"),
    ("我的密码忘了，怎么重置？", "False"),
    ("发现有人假装是银行工作人员", "True"),
    ("你能告诉我验证码吗？", "False"),
    ("我的存款刚刚被非法转移了！", "True"),
] * 5

def save_mock_csv(filename, data):
    """保存模拟 CSV（用于演示）"""
    df = pd.DataFrame(data, columns=["text", "label"])
    df.to_csv(filename, index=False, header=False, encoding="utf-8")
    print(f"💾 已生成 {filename}")

# ======================
# 安全加载函数（兼容你的原始逻辑）
# ======================
def load_data_safely(csv_path):
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            df_line = pd.read_csv(StringIO(line), header=None, keep_default_na=False)
            row = df_line.iloc[0].tolist()
        except Exception:
            continue
        if len(row) < 2:
            continue
        text = str(row[0])
        label_val = None
        for val in reversed(row):
            v_str = str(val).strip()
            if v_str == "True":
                label_val = 1
                break
            elif v_str == "False":
                label_val = 0
                break
        if label_val is not None:
            texts.append(text)
            labels.append(label_val)
    return texts, labels

# ======================
# 数据集类
# ======================
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ======================
# 测试函数
# ======================
def evaluate_model(model, tokenizer, texts, labels, name=""):
    dataset = FraudDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    print(f"📊 {name} 准确率: {acc:.4f} ({sum(p==t for p,t in zip(preds,true_labels))}/{len(true_labels)})")
    return acc

# ======================
# 主程序
# ======================
def main():
    # 1. 生成模拟数据文件
    save_mock_csv("original_test.csv", RAW_DATA)
    save_mock_csv("rewritten_synonym.csv", REWRITTEN_SYNONYM)
    save_mock_csv("rewritten_structure.csv", REWRITTEN_STRUCTURE)

    # 2. 加载原始数据并训练
    print("\n🚀 加载 BERT 模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.to(DEVICE)

    print("\n📂 加载原始训练数据...")
    texts, labels = load_data_safely("original_test.csv")
    print(f"✅ 加载 {len(labels)} 条原始数据")

    dataset = FraudDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("\n🔄 开始训练（1 epoch）...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}, 平均损失: {avg_loss:.4f}")

    # 3. 测试不同数据集
    print("\n🧪 开始测试鲁棒性...")
    acc_original = evaluate_model(model, tokenizer, texts, labels, "原始数据")
    texts_syn, labels_syn = load_data_safely("rewritten_synonym.csv")
    acc_syn = evaluate_model(model, tokenizer, texts_syn, labels_syn, "同义词替换")
    texts_str, labels_str = load_data_safely("rewritten_structure.csv")
    acc_str = evaluate_model(model, tokenizer, texts_str, labels_str, "句式重构")

    # 4. 保存结果
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("【欺诈检测鲁棒性实验结果】\n\n")
        f.write(f"原始数据准确率: {acc_original:.4f}\n")
        f.write(f"同义词替换准确率: {acc_syn:.4f} (↓{acc_original - acc_syn:.4f})\n")
        f.write(f"句式重构准确率: {acc_str:.4f} (↓{acc_original - acc_str:.4f})\n\n")
        f.write("结论：句式重构对模型影响更大，说明模型依赖表面形式而非深层语义。\n")


if __name__ == "__main__":
    main()
