import re
import os
import csv
import json
import psutil
import numpy as np
import tkinter as tk
from tkinter import filedialog
from llama_cpp import Llama
from environment_variables import model_path_gpt, model_path_qwen, LlmParameter

# ---------------------------------------------------------
# UI 選擇 txt 檔案
# ---------------------------------------------------------
print("請在彈出的視窗中選擇你要分析的 txt 文本...")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="選擇要分析的文學文本",
    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
)

if not file_path:
    print("❌ 你沒有選擇檔案，程式結束。")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    full_document_text = f.read()

sentences = [s.strip() for s in re.split(r'(?<=[\n])', full_document_text) if s.strip()]

# ---------------------------------------------------------
# 提前定義輸出檔案名稱 (為了斷點續傳)
# ---------------------------------------------------------
parameter = LlmParameter(model_type='GPT')
[fp, fn] = os.path.split(file_path)
fn = os.path.splitext(fn)[0]
[mp, mn] = os.path.split(parameter.model_path)
mn = os.path.splitext(mn)[0]

output_filename = f"探測結果_v3.1_{fn}_{mn}.csv"
pass1_cache_filename = f"探測結果_v3.1_{fn}_{mn}_pass1.jsonl"

# ---------------------------------------------------------
# 斷點續傳檢查機制
# ---------------------------------------------------------
completed_csv_rows = 0
completed_pass1_rows = 0
contextual_results = []

print("\n" + "=" * 50)
print("🔍 檢查中斷紀錄 (斷點續傳)...")
# 1. 檢查最終 CSV 進度
if os.path.exists(output_filename):
    with open(output_filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        lines = list(reader)
        if len(lines) > 1:  # 扣除標題行
            completed_csv_rows = len(lines) - 1
    print(f"  ✅ 偵測到 CSV 紀錄，已完成 {completed_csv_rows}/{len(sentences)} 句最終輸出。")

# 2. 檢查第一階段 (脈絡模式) 暫存進度
if os.path.exists(pass1_cache_filename):
    with open(pass1_cache_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                contextual_results.append(json.loads(line))
    completed_pass1_rows = len(contextual_results)
    print(f"  ✅ 偵測到脈絡模式暫存，已儲存 {completed_pass1_rows}/{len(sentences)} 句脈絡機率。")

if completed_csv_rows >= len(sentences):
    print("文本已完全處理完畢，任務結束！")
    exit()

# ---------------------------------------------------------
# 載入模型
# ---------------------------------------------------------
print("=" * 50)
print("正在載入 GGUF 模型...")
MAX_CONTEXT_WINDOW = 10240

llm = Llama(
    model_path=parameter.model_path,
    n_gpu_layers=parameter.n_gpu_layers,
    n_threads=parameter.n_threads,
    n_ctx=MAX_CONTEXT_WINDOW,
    logits_all=True,
    verbose=False,
    n_batch=512,
    flash_attn=True
)

# ---------------------------------------------------------
# Token 預算安全檢查
# ---------------------------------------------------------
full_tokens = llm.tokenize(full_document_text.encode('utf-8'))
total_token_count = len(full_tokens)

print(f"\n輸入文本總 Token 數: {total_token_count:,} tokens")
if total_token_count >= MAX_CONTEXT_WINDOW:
    print("⚠️【警告】文本長度已超過模型的 Context Window！建議中斷。")
    if input("請問是否要繼續強制執行？(y/n): ").lower() != 'y':
        exit()


# ---------------------------------------------------------
# 實時監控函數
# ---------------------------------------------------------
def report_hardware_status(model_file_path, current_token_count):
    vm = psutil.virtual_memory()
    ram_usage_percent = vm.percent
    ram_used_gb = vm.used / (1024 ** 3)
    model_size_gb = os.path.getsize(model_file_path) / (1024 ** 3)
    kv_cache_gb = (current_token_count * 2 * 28 * 4 * 128 * 2) / (1024 ** 3)
    estimated_vram_gb = model_size_gb + kv_cache_gb
    print(f"      [監控] RAM: {ram_used_gb:.1f}GB ({ram_usage_percent}%) | 推估 VRAM: {estimated_vram_gb:.1f}GB")


def calculate_logprob(logits, token_id):
    max_logit = np.max(logits)
    log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
    return float(logits[token_id] - log_sum_exp)


# ---------------------------------------------------------
# 第一階段：脈絡模式 (含 KV Cache 光速重建)
# ---------------------------------------------------------
if completed_csv_rows < len(sentences):  # 如果 CSV 沒寫完，代表我們需要第一階段的資料
    print("\n" + "=" * 50)
    print(f"第一階段：連續執行【脈絡模式】預測")
    print("=" * 50)

    accumulated_tokens = 0
    llm.reset()

    # ⏩ 光速重建 KV Cache (如果之前有斷點)
    if 0 < completed_pass1_rows < len(sentences):
        print(f"正在快速重建 KV Cache 至第 {completed_pass1_rows} 句 (不計算機率，僅恢復記憶)...")
        past_tokens = []
        for i in range(completed_pass1_rows):
            t = llm.tokenize(sentences[i].encode('utf-8'))
            if t: past_tokens.extend(t)

        # 按照 n_batch 批次推入模型重建 Cache
        for i in range(0, len(past_tokens), llm.n_batch):
            llm.eval(past_tokens[i:i + llm.n_batch])
            accumulated_tokens += len(past_tokens[i:i + llm.n_batch])
        print("✅ KV Cache 重建完成，無縫接軌繼續推論！")

    # 以 Append 模式開啟 JSONL 暫存檔
    pass1_file = open(pass1_cache_filename, 'a', encoding='utf-8')

    for i, sentence in enumerate(sentences):
        if i < completed_pass1_rows:
            continue  # 已經算過的直接跳過

        tokens = llm.tokenize(sentence.encode('utf-8'))
        if not tokens:
            result_item = [0.0, []]
            contextual_results.append(result_item)
            pass1_file.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            pass1_file.flush()
            os.fsync(pass1_file.fileno())
            continue

        target_probs = []
        token_details = []
        is_first = True

        for token in tokens:
            accumulated_tokens += 1
            if is_first:
                llm.eval([token])
                is_first = False
                continue

            logits = np.array(llm.scores[llm.n_tokens - 1])
            lp = calculate_logprob(logits, token)
            target_probs.append(lp)
            token_str = llm.detokenize([token]).decode('utf-8', errors='ignore')
            token_details.append({token_str: round(lp, 4)})

            llm.eval([token])

        avg_prob = sum(target_probs) / len(target_probs) if target_probs else 0.0
        result_item = [avg_prob, token_details]
        contextual_results.append(result_item)

        # 實時寫入磁區，防範藍屏
        pass1_file.write(json.dumps(result_item, ensure_ascii=False) + '\n')
        pass1_file.flush()
        os.fsync(pass1_file.fileno())

        print(f"  ⚡ 脈絡預測 ({i + 1}/{len(sentences)}) | 機率: {avg_prob:.4f}")
        if (i + 1) % 5 == 0 or (i + 1) == len(sentences):
            report_hardware_status(model_path, accumulated_tokens)

    pass1_file.close()

# ---------------------------------------------------------
# 第二階段：孤立模式 (實時寫入最終 CSV)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print(f"第二階段：連續執行【孤立模式】並實時寫入 CSV")
print("=" * 50)

csv_headers = ["句子序號", "文本內容", "孤立總平均機率", "脈絡總平均機率", "上下文紅利 (Style Score)",
               "孤立單字機率分布", "脈絡單字機率分布"]
file_mode = 'a' if os.path.exists(output_filename) else 'w'

with open(output_filename, mode=file_mode, encoding='utf-8-sig', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if file_mode == 'w':
        writer.writeheader()
        csv_file.flush()

    for i, sentence in enumerate(sentences):
        if i < completed_csv_rows:
            continue  # 已經在 CSV 裡的句子直接跳過

        llm.reset()  # 孤立模式每次都要重置 Cache
        tokens = llm.tokenize(sentence.encode('utf-8'))

        target_probs = []
        isolated_details = []
        is_first = True

        if tokens:
            for token in tokens:
                if is_first:
                    llm.eval([token])
                    is_first = False
                    continue

                logits = np.array(llm.scores[llm.n_tokens - 1])
                lp = calculate_logprob(logits, token)
                target_probs.append(lp)

                token_str = llm.detokenize([token]).decode('utf-8', errors='ignore')
                isolated_details.append({token_str: round(lp, 4)})
                llm.eval([token])

        isolated_logprob = sum(target_probs) / len(target_probs) if target_probs else 0.0
        print(f"  🌱 孤立預測 ({i + 1}/{len(sentences)}) | 機率: {isolated_logprob:.4f}")

        # 取出第一階段紀錄，計算紅利
        contextual_logprob, contextual_details = contextual_results[i]
        style_score = contextual_logprob - isolated_logprob

        writer.writerow({
            "句子序號": i + 1,
            "文本內容": sentence,
            "孤立總平均機率": round(isolated_logprob, 4),
            "脈絡總平均機率": round(contextual_logprob, 4),
            "上下文紅利 (Style Score)": round(style_score, 4),
            "孤立單字機率分布": json.dumps(isolated_details, ensure_ascii=False),
            "脈絡單字機率分布": json.dumps(contextual_details, ensure_ascii=False)
        })

        # 強制將資料從作業系統緩衝區刷入物理磁碟
        csv_file.flush()
        os.fsync(csv_file.fileno())

print(f"\n  全部完成！最終數據已安全儲存至: {os.path.abspath(output_filename)}")