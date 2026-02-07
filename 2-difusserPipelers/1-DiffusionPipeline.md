
## ¿Qué es DiffusionPipeline?
Es una herramienta que combina varios modelos de inteligencia artificial (como UNET, codificadores de texto, etc.) en una sola interfaz fácil de usar para generar imágenes.

## Cómo cargar un pipeline

### Método 1: Usar la clase general
```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
```

### Método 2: Usar la clase específica
```python
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
```

## Tipos de pipelines disponibles

| Pipeline | Función |
|----------|---------|
| QwenImagePipeline | Texto a imagen |
| QwenImageImg2ImgPipeline | Imagen a imagen |
| QwenImageInpaintPipeline | Relleno de imágenes |

## Cómo usar modelos locales

### Paso 1: Descargar el modelo
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen-Image")
```

### Paso 2: Cargar desde tu computadora
```python
pipeline = QwenImagePipeline.from_pretrained(
    "ruta/a/tu/cache", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
```

## Control de precisión
Puedes cargar modelos con diferentes niveles de precisión para ahorrar memoria:

```python
# Todos los modelos en bfloat16
pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16
)

# Modelos específicos con diferente precisión
pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype={"transformer": torch.bfloat16, "default": torch.float16}
)
```

## Configuración de dispositivos

### Opciones para device_map:
- **"cuda"**: Usa la GPU
- **"balanced"**: Distribuye entre varias GPUs

```python
pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

## Carga rápida para modelos grandes
```python
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

pipeline = DiffusionPipeline.from_pretrained(
    "modelo-grande", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
```

## Personalización avanzada

### Cambiar componentes
```python
from diffusers import AutoModel

# Usar un VAE mejorado
vae = AutoModel.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

### Reutilizar modelos entre pipelines
```python
from diffusers import AutoPipelineForText2Image

# Primer pipeline
pipeline_sdxl = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    device_map="cuda"
)

# Segundo pipeline que reusa modelos
pipeline = AutoPipelineForText2Image.from_pipe(
    pipeline_sdxl, 
    enable_pag=True
)
```

## Seguridad
Para desactivar el filtro de contenido (no recomendado para aplicaciones públicas):
```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    safety_checker=None
)
```

-------------------------------------------
-------------------------------------------
-------------------------------------------
-------------------------------------------


Aquí tienes el **tutorial en español** sobre **DiffusionPipeline en Diffusers** explicado de forma clara y práctica.

---

# 🧠 ¿Qué es `DiffusionPipeline`?

Los modelos de difusión están formados por varios componentes:

* UNet o DiT (modelo de difusión)
* Text Encoder
* VAE (autoencoder)
* Scheduler

`DiffusionPipeline` **envuelve todos esos componentes en una sola API fácil de usar**, pero sin quitarte la posibilidad de modificarlos individualmente.

Es la forma **oficial y moderna** de cargar modelos en Diffusers.

---

# 🚀 Cargar un pipeline

`DiffusionPipeline` detecta automáticamente qué clase de pipeline usar leyendo el archivo `model_index.json` del modelo.

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

---

## 🧩 Usar la subclase específica del pipeline

Cada modelo tiene subclases especializadas:

| Subclase                   | Tarea           |
| -------------------------- | --------------- |
| `QwenImagePipeline`        | texto → imagen  |
| `QwenImageImg2ImgPipeline` | imagen → imagen |
| `QwenImageInpaintPipeline` | inpainting      |

Puedes cargarlas directamente:

```python
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

---

# 💾 Ejecutar el modelo localmente (sin volver a descargar)

Descarga el modelo al caché:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen-Image")
```

Luego cárgalo desde la ruta local:

```python
pipeline = QwenImagePipeline.from_pretrained(
    "ruta/a/tu/cache",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

---

# 🎚️ Tipos de datos (`torch_dtype`)

Reducir la precisión baja el consumo de VRAM.

### Un solo tipo para todo:

```python
pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16
)
```

### Diferente precisión por componente:

```python
pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype={
        "transformer": torch.bfloat16,
        "default": torch.float16
    }
)
```

---

# 🖥️ Ubicación en dispositivos (`device_map`)

| Opción       | Descripción               |
| ------------ | ------------------------- |
| `"cuda"`     | Coloca todo en GPU        |
| `"balanced"` | Distribuye en varias GPUs |

```python
pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

Ver dónde quedó cada parte:

```python
print(pipeline.hf_device_map)
```

---

## 🔄 Resetear `device_map`

Necesario si usarás:

* `.to()`
* `enable_model_cpu_offload()`
* `enable_sequential_cpu_offload()`

```python
pipeline.reset_device_map()
```

---

# ⚡ Carga paralela (MUY IMPORTANTE para modelos grandes)

Acelera muchísimo la carga.

```python
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

---

# 🔁 Reemplazar partes del pipeline (ej: VAE mejorado)

```python
from diffusers import DiffusionPipeline, AutoModel

vae = AutoModel.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

---

# ♻️ Reutilizar modelos en varios pipelines (ahorra VRAM)

```python
from diffusers import AutoPipelineForText2Image

pipeline_sdxl = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Crear otro pipeline usando los mismos pesos
pipeline = AutoPipelineForText2Image.from_pipe(
    pipeline_sdxl,
    enable_pag=True
)
```

👉 Ambos usan **la misma memoria**.

⚠️ Comparten estado: si modificas uno, afecta al otro.

---

# 🛡️ Safety Checker (filtro de contenido)

Puedes desactivarlo (no recomendado en apps públicas):

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None
)
```

---

# 🧠 Lo más importante que debes recordar

1. **Siempre usa `DiffusionPipeline` o `AutoPipeline`**
2. Usa `torch_dtype` para ahorrar VRAM
3. Usa `device_map="cuda"`
4. Activa `HF_ENABLE_PARALLEL_LOADING`
5. Puedes cambiar VAE, schedulers, etc.
6. Puedes reutilizar modelos sin gastar más memoria

---

Si quieres, en el siguiente mensaje puedo darte el **template profesional base para Colab** que sirve para **cualquier modelo Diffusers (imagen, video, I2V, T2V, inpaint, etc.)** sin que vuelvas a tener errores de versiones.
