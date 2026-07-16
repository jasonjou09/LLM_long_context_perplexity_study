# This file is for storing universal path locations. You can change them here.

# 這是 Intel oneAPI 編譯器 DLL 的預設安裝路徑
intel_compiler_bin = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
intel_mkl_bin = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin"

# 模型路徑 (改成你自己的GGUF檔案的位置)
model_path_gpt = r"E:\LM Studio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
model_path_qwen3_bs = r"E:\LM Studio\models\mradermacher\Qwen3-4B-Base-GGUF\Qwen3-4B-Base.Q4_K_S.gguf"
model_path_qwen3_5_bs = r"E:\LM Studio\models\titan0115\Qwen3.5-4B-Base\Qwen3.5-4B-Base-Q4_K_S.gguf"
model_path_qwen3_5 = r"E:\LM Studio\models\HauhauCS\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

class LlmParameter:
    def __init__(self, model_type, model_path = "", n_gpu_layers = 0, n_threads = 0, n_ctx = 0):
        """model_type: string ('GPT', 'Qwen3Base', 'Qwen3.5Base', 'Qwen3.5' or anything other string, you must specify it)"""

        self.model_type = model_type

        if self.model_type == 'GPT':
            self.model_path = model_path_gpt
            self.n_gpu_layers = 17 if n_gpu_layers == 0 else n_gpu_layers
            self.n_threads = 6 if n_threads == 0 else n_threads
            self.n_ctx = 8192 if n_ctx == 0 else n_ctx

        elif self.model_type == 'Qwen3Base':
            self.model_path = model_path_qwen3_bs
            self.n_gpu_layers = 36 if n_gpu_layers == 0 else n_gpu_layers
            self.n_threads = 6 if n_threads == 0 else n_threads
            self.n_ctx = 16384 if n_ctx == 0 else n_ctx

        elif self.model_type == 'Qwen3.5Base':
            self.model_path = model_path_qwen3_5_bs
            self.n_gpu_layers = 36 if n_gpu_layers == 0 else n_gpu_layers
            self.n_threads = 6 if n_threads == 0 else n_threads
            self.n_ctx = 16384 if n_ctx == 0 else n_ctx

        elif self.model_type == 'Qwen3.5':
            self.model_path = model_path_qwen3_5
            self.n_gpu_layers = 36 if n_gpu_layers == 0 else n_gpu_layers
            self.n_threads = 6 if n_threads == 0 else n_threads
            self.n_ctx = 16384 if n_ctx == 0 else n_ctx

        elif model_path == "" or n_gpu_layers == 0 or n_threads == 0:
            print("model type unrecognized while detailed parameter unspecified, defaulting to Qwen3.5 Instruct")
            self.model_path = model_path_qwen3_5
            self.n_gpu_layers = 36
            self.n_threads = 6
            self.n_ctx = 16384