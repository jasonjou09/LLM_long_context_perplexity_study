import os
import csv
import time
from datetime import datetime
from environment_variables import intel_compiler_bin, intel_mkl_bin, model_path_gpt, model_path_qwen

# 針對 Intel Arc GPU (SYCL) 的環境變數注入
if os.name == 'nt':
    for path in [intel_compiler_bin, intel_mkl_bin]:
        if os.path.exists(path):
            os.add_dll_directory(path)  # Python 3.8+ 專用安全載入機制
            os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")
from llama_cpp import Llama

# ---------------------------------------------------------
# 模型載入設定
# ---------------------------------------------------------
# 請換成你實際要測試的大模型路徑
model_path = model_path_qwen

print("=" * 50)
print("正在將模型載入 SYCL (Intel Arc) 後端...")
# 注意：移除了 logits_all=True，因為一般對話不需要計算所有備胎詞的機率
llm = Llama(
    model_path=model_path,
    n_gpu_layers=16,
    n_threads=6,
    n_ctx=8192,
    verbose=False,  # 關閉底層 C++ 日誌以保持終端機乾淨
    flash_attn=True,
    type_k=8,
    type_v=8,
)
print("✅ 模型載入完成！")
print("=" * 50)

# ---------------------------------------------------------
# 對話與紀錄系統初始化
# ---------------------------------------------------------
chat_history = []  # 儲存給模型看的完整對話上下文
csv_export_data = []  # 儲存要寫入 CSV 的效能分析數據

print("進入對話模式。")
print("💡 提示：輸入 'exit' 或 'quit' 即可結束對話，並自動匯出效能 CSV 紀錄檔。\n")

while True:
    # 1. 取得使用者輸入
    user_input = input("👤 User: ")

    # 檢查是否觸發離開指令
    if user_input.strip().lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue

    # 將使用者輸入加入上下文記憶
    chat_history.append({"role": "user", "content": user_input})

    print("Model : ", end="", flush=True)

    # 2. 效能追蹤變數初始化
    start_time = time.time()
    generated_text = ""
    generated_tokens = 0

    # 3. 呼叫 Chat Completion API 並開啟串流 (Streaming)
    stream = llm.create_chat_completion(
        messages=chat_history,
        stream=True,  # 逐字吐出，模擬真實打字體驗
        max_tokens=4096  # 設定單次生成最大上限
    )

    # 4. 接收串流並實時計算速度
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            token_str = chunk["choices"][0]["delta"]["content"]
            print(token_str, end="", flush=True)
            generated_text += token_str
            generated_tokens += 1

    # 5. 結算單次生成效能
    end_time = time.time()
    time_taken = end_time - start_time
    # 避免除以零的極端情況
    speed_tps = generated_tokens / time_taken if time_taken > 0 else 0.0

    # 將 AI 的回覆加入上下文記憶
    chat_history.append({"role": "assistant", "content": generated_text})

    # 6. 計算當前對話總 Token 使用量 (包含歷史紀錄)
    # 利用模型的 tokenize 功能粗估當前上下文長度
    history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    total_context_tokens = len(llm.tokenize(history_string.encode('utf-8')))

    # 7. 顯示效能報告
    print(
        f"\n\n   ⚡ [SYCL 效能監控] 速度: {speed_tps:.2f} tokens/s | 本次生成: {generated_tokens} tokens | 累積上下文: {total_context_tokens} tokens\n")
    print("-" * 50)

    # 紀錄至匯出列表
    csv_export_data.append({
        "對話輪次": len(csv_export_data) + 1,
        "User輸入": user_input,
        "AI輸出": generated_text,
        "生成速度 (tokens/s)": round(speed_tps, 2),
        "單次回覆量 (tokens)": generated_tokens,
        "累積上下文用量 (tokens)": total_context_tokens
    })

# ---------------------------------------------------------
# 結束程式與匯出 CSV
# ---------------------------------------------------------
if csv_export_data:
    print("\n" + "=" * 50)
    print("💾 正在匯出對話與效能紀錄...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"SYCL_測速對話紀錄_{timestamp}.csv"

    csv_headers = ["對話輪次", "User輸入", "AI輸出", "生成速度 (tokens/s)", "單次回覆量 (tokens)",
                   "累積上下文用量 (tokens)"]

    with open(output_filename, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_export_data)

    print(f"    紀錄已成功儲存至: {os.path.abspath(output_filename)}")
else:
    print("\n沒有任何對話紀錄，程式結束。")