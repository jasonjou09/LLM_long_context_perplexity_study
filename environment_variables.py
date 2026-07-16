# This file is for storing universal path locations. You can change them here.

# 這是 Intel oneAPI 編譯器 DLL 的預設安裝路徑
intel_compiler_bin = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
intel_mkl_bin = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin"

# 模型路徑 (改成你自己的GGUF檔案的位置)
model_path_gpt = r"E:\LM Studio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
model_path_qwen = r"E:\LM Studio\models\mradermacher\Qwen3-4B-Base-GGUF\Qwen3-4B-Base.Q4_K_S.gguf"

class LlmParameter:
    def __init__(self, model_type, model_path, n_gpu_layers, n_threads):
        """model_type: string ('GPT' or 'Qwen')"""

        self.model_type = model_type

        if self.model_type == 'GPT':
            self.model_path = model_path_gpt
            self.n_gpu_layers = 17
            self.n_threads = 6

        elif self.model_type == 'Qwen':
            self.model_path = model_path_qwen
            self.n_gpu_layers = 36
            self.n_threads = 6
        else:
            print("model type not specified, defaulting to Qwen")
            self.model_path = model_path_qwen
            self.n_gpu_layers = 36
            self.n_threads = 6
