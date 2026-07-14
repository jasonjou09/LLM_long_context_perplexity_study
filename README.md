# LLM_long_context_perplexity_study
This repo aims to evaluate a LLM's performance by evaluating its reduction in perplexity for a given word when entire long context is provided compared to only current sentence being provided.
It provides an alternative way to view how well a model learns a long context document like novels, and gives measureble metrics to evaluate them.


This readme provides a step-by-step guide to installing and configuring `llama-cpp-python` with hardware acceleration (Vulkan, CUDA, or Intel SYCL) on Windows. 

---

## Table of Contents
1. [Prerequisites (C++ Build Environment)](#1-prerequisites-c-build-environment)
2. [Backend Options](#2-backend-options)
   - [Option A: Vulkan Backend (Universal GPU)](#option-a-vulkan-backend-universal-gpu)
   - [Option B: CUDA Backend (NVIDIA GPU)](#option-b-cuda-backend-nvidia-gpu)
   - [Option C: SYCL Backend (Intel Arc / Integrated GPU)](#option-c-sycl-backend-intel-arc--integrated-gpu)
3. [ Shell Syntax Configuration Tips](#-shell-syntax-configuration-tips)

---

## 1. Prerequisites (C++ Build Environment)

Before compiling `llama-cpp-python` with hardware acceleration, you must set up a C++ compilation environment because `cmake` requires it to build the native bindings.

1. Go to the [Visual Studio Downloads page](https://visualstudio.microsoft.com/downloads/).
2. Scroll down to **All Downloads** > **Tools for Visual Studio** > and download **Build Tools for Visual Studio 2026**.
   
   ![Build Tools for Visual Studio](./assets/vs_build_tools_download.png)

3. Run the installer, select **Build Tools for Visual Studio 2026**, and install.

   ![VS Build Tools Install Option](./assets/vs_build_tools_install.png)

   > ⚠️ **WARNING:** Do **NOT** install the full Visual Studio IDE. You only need the **compiler (Build Tools)** for python/cmake bindings.

---

## 2. Backend Options

Choose **ONE** of the acceleration backends below depending on your system's GPU hardware.

---

### Option A: Vulkan Backend (Universal GPU)
*Compatible with AMD, Intel, and NVIDIA GPUs supporting Vulkan.*

#### Step A.1: Install Vulkan SDK
1. Go to the [LunarG Vulkan SDK Website](https://vulkan.lunarg.com/sdk/home).
2. Download the first **Windows (x64 / x86)** installer.
   
   ![Vulkan SDK Download](./assets/vulkan_sdk_download.png)

3. Complete the installation with the default settings.
4. **Important:** If your IDE (e.g., PyCharm) was open during the installation, **you must restart the IDE completely** for the Vulkan environment variables to take effect.

#### Step A.2: Compile & Install in PyCharm Terminal
1. Open PyCharm, and first install the necessary python build tools:
   ```bash
   pip install cmake psutil scikit-build-core

```

2. In PyCharm's built-in **PowerShell Terminal**, run the following:
```powershell
# 1. Verify Vulkan SDK is correctly loaded
echo $env:VULKAN_SDK

# 2. Configure CMake and compile llama-cpp-python
$env:CMAKE_ARGS="-DGGML_VULKAN=1"
pip install llama-cpp-python --no-cache-dir

```



---

### Option B: CUDA Backend (NVIDIA GPU)

*Best performance for NVIDIA RTX / GTX graphic cards.*

#### Step B.1: Install CUDA Toolkit

1. Visit the [NVIDIA CUDA Toolkit Archive](https://www.google.com/search?q=https://developer.nvidia.com/cuda-toolkit-archive).
2. Select and download **CUDA Toolkit 12.4.1** (recommended for stability and broad framework compatibility).
3. Install using all default options.

#### Step B.2: Compile & Install in PyCharm Terminal

1. Open PyCharm, and first install build dependencies:
```bash
pip install cmake psutil scikit-build-core

```


2. In PyCharm's built-in **PowerShell Terminal**, check for CUDA availability:
```powershell
# 1. Verify CUDA environment path
echo $env:CUDA_PATH

```


3. Compile and pull the prebuilt/source compiler wheel for CUDA 12.4:
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu124](https://abetlen.github.io/llama-cpp-python/whl/cu124)

```



---

### Option C: SYCL Backend (Intel Arc / Integrated GPU)

*Optimized for Intel Arc discrete graphics and Intel Core Ultra integrated graphics.*

#### Step C.1: Install Intel oneAPI Base Toolkit

1. Go to the [Intel oneAPI Base Toolkit Page](https://www.google.com/search?q=https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html).
2. Click **Get it now** and download the installer (the **Offline installer** is recommended).
3. Run the installer and select **Custom Installation** to save disk space. Select only:
* **Intel DPC++ Compiler** / **Intel DPC++/C++ Compiler** (`icx` compiler engine)
* **Intel oneAPI Math Kernel Library (oneMKL)**


4. During installation, when prompted to integrate or register with Visual Studio, check the box to link with your installed VS Build Tools.

#### Step C.2: Compile via VS Command Prompt (CMD)

*Note: Intel oneAPI configuration environment must be compiled inside the Native VS Command Prompt.*

1. Open **"x64 Native Tools Command Prompt for VS"** from your Windows search bar.
2. Load the oneAPI compiler environment variables (Only available in CMD):
```cmd
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

```


3. Navigate to your project virtual environment and activate it:
```cmd
cd C:\\Users\\[your_project_path]\\.venv\\Scripts
activate

```


4. Install **Ninja** build system and compile the package:
```cmd
pip install ninja
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
pip install llama-cpp-python --no-cache-dir --force-reinstall

```



#### Step C.3: Required Python Runtime Setup

Because Intel oneAPI relies on dynamically linked libraries (`.dll`), Windows must know where to find them at runtime. You **MUST** add these lines in your python code **before importing `llama_cpp**`:

```python
import os

if os.name == 'nt':
    # Default paths for Intel compiler and MKL libraries
    intel_compiler_bin = r"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin"
    intel_mkl_bin = r"C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\bin"
    
    for path in [intel_compiler_bin, intel_mkl_bin]:
        if os.path.exists(path):
            os.add_dll_directory(path)  # Secure DLL loading (Python 3.8+)
            os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")

# Now import llama-cpp-python safely
import llama_cpp

```

---

## 💡 Shell Syntax Configuration Tips

When configuring environment variables in Windows, the syntax varies significantly based on the shell/terminal emulator you are using:

| Environment Variable Task | PowerShell (PyCharm Default) | CMD (Command Prompt / VS Tools) |
| --- | --- | --- |
| **Set variable** | `$env:CMAKE_ARGS="-DGGML_VULKAN=1"` | `set CMAKE_ARGS=-DGGML_VULKAN=1` |
| **Check variable value** | `echo $env:VULKAN_SDK` | `echo %VULKAN_SDK%` |

If you run into previous compiler caching errors, append `--no-cache-dir --force-reinstall` to your pip command to ensure a clean build.
