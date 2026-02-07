Aquí tienes el **tutorial en español sobre Community Pipelines y Community Components en Diffusers**, explicado de forma práctica.

---

# 🌍 ¿Qué son los *Community Pipelines*?

Son pipelines que **no vienen en la implementación oficial del paper**, sino que:

* Añaden nuevas funciones
* Extienden pipelines existentes
* Permiten tareas nuevas (ej: pix2pix especial, video, LPW prompts, etc.)

Pueden estar en:

| Ubicación            | Característica                                             |
| -------------------- | ---------------------------------------------------------- |
| **GitHub Diffusers** | Revisados por el equipo Diffusers (más lento, más oficial) |
| **Hugging Face Hub** | Subidos libremente por usuarios (más rápido, más flexible) |

👉 En uso práctico, **se cargan exactamente igual**.

---

# 🧩 `custom_pipeline`

Para cargar un community pipeline se usa el parámetro:

```python
custom_pipeline="nombre_del_pipeline"
```

Ejemplo:

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="pipeline_stable_diffusion_3_instruct_pix2pix",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

---

## 🕒 Cargar una versión específica

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="pipeline_stable_diffusion_3_instruct_pix2pix",
    custom_revision="main",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

---

## ⚠️ Seguridad

Aunque el Hub escanea archivos:

> Siempre revisa el código del pipeline si usas community pipelines.

---

# 📁 Cargar un pipeline comunitario LOCAL

Si tienes una carpeta con `pipeline.py`:

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="ruta/a/tu/carpeta",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

---

# ♻️ Usar `from_pipe()` con community pipelines (ahorra VRAM)

Esto es muy potente.

```python
pipeline_sd = DiffusionPipeline.from_pretrained(
    "emilianJR/CyberRealistic_V3",
    torch_dtype=torch.float16,
    device_map="cuda"
)

pipeline_lpw = DiffusionPipeline.from_pipe(
    pipeline_sd,
    custom_pipeline="lpw_stable_diffusion",
    device_map="cuda"
)
```

👉 No carga pesos nuevos.
👉 Solo añade la nueva funcionalidad.

Muchos community pipelines **no tienen pesos**, solo agregan capacidades.

---

# 🧱 Community Components (nivel avanzado)

Permite usar:

* UNet personalizados
* VAEs personalizados
* Transformers personalizados
* Schedulers no soportados oficialmente

Necesitas:

* Archivos `.py` con las clases
* Respetar la estructura esperada por Diffusers

---

## 🧪 Ejemplo real: `showlab/show-1-base` (Texto → Video)

### 1️⃣ Cargar componentes base

```python
from transformers import T5Tokenizer, T5EncoderModel, CLIPImageProcessor
from diffusers import DPMSolverMultistepScheduler

pipeline_id = "showlab/show-1-base"

tokenizer = T5Tokenizer.from_pretrained(pipeline_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipeline_id, subfolder="text_encoder")
scheduler = DPMSolverMultistepScheduler.from_pretrained(pipeline_id, subfolder="scheduler")
feature_extractor = CLIPImageProcessor.from_pretrained(pipeline_id, subfolder="feature_extractor")
```

---

### 2️⃣ Cargar un UNet personalizado

Archivo: `showone_unet_3d_condition.py`

```python
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(
    pipeline_id,
    subfolder="unet"
)
```

---

### 3️⃣ Cargar el pipeline personalizado

Archivo: `pipeline_t2v_base_pixel.py`

```python
from pipeline_t2v_base_pixel import TextToVideoIFPipeline

pipeline = TextToVideoIFPipeline(
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
    device_map="cuda",
    torch_dtype=torch.float16
)
```

---

### 4️⃣ Subir al Hub

```python
pipeline.push_to_hub("custom-t2v-pipeline")
```

Luego debes:

* Editar `model_index.json`
* Subir los `.py` al repo
* Ajustar `_class_name`

---

# 🔐 `trust_remote_code=True`

Para ejecutar estos pipelines:

```python
pipeline = DiffusionPipeline.from_pretrained(
    "tu-repo",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
```

⚠️ Recomendación profesional:

```python
revision="commit_hash"
```

Para evitar que el código cambie por algo malicioso.

---

# 🧠 Idea clave

Community pipelines sirven para:

* LPW prompts
* Pix2Pix avanzados
* Video pipelines experimentales
* Funciones que Diffusers aún no soporta
* Extender modelos sin recargar pesos

---

# 🏆 Resumen profesional

| Necesidad                     | Solución                 |
| ----------------------------- | ------------------------ |
| Pipeline especial no oficial  | `custom_pipeline`        |
| Añadir funciones sin más VRAM | `from_pipe()`            |
| Usar UNet/VAE propios         | Community Components     |
| Ejecutar código externo       | `trust_remote_code=True` |

---

¡Excelente! Ahora vamos a profundizar en los **Community Pipelines** (Pipelines de la Comunidad) y los **Componentes Comunitarios**.

---

## Pipelines y Componentes de la Comunidad

Los **Community Pipelines** son clases basadas en `DiffusionPipeline` que difieren de la implementación original de los artículos de investigación. Estas versiones extienden la funcionalidad original o añaden características completamente nuevas.

### GitHub vs. Hub: ¿Dónde se guardan?

Existen dos lugares principales donde puedes encontrar o subir estos pipelines:

| Característica | **GitHub** (Repositorio de Diffusers) | **Hub** (Hugging Face) |
| --- | --- | --- |
| **Uso** | Se usa el nombre del archivo del pipeline. | Se usa el ID del repositorio. |
| **Revisión** | Requiere un *Pull Request* y revisión manual del equipo de Diffusers (más lento). | Subida directa sin revisión previa (más rápido). |
| **Visibilidad** | Aparece en la documentación oficial. | Aparece en tu perfil y depende de tu promoción. |

---

## Cómo usar `custom_pipeline`

Para cargar cualquiera de estos tipos de pipelines, utiliza el argumento `custom_pipeline` en el método `from_pretrained()`.

### Ejemplo básico:

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="pipeline_stable_diffusion_3_instruct_pix2pix",
    torch_dtype=torch.float16,
    device_map="cuda"
)

```

### Otras formas de carga:

* **Desde una carpeta local:** Si tienes un archivo `pipeline.py` en una carpeta, pasa la ruta de la carpeta a `custom_pipeline`.
* **Desde otro pipeline (`from_pipe`):** Ideal para ahorrar memoria. Si ya tienes un pipeline cargado, puedes crear uno comunitario reutilizando los mismos componentes:
```python
# Reutiliza los pesos de un modelo ya cargado para un pipeline de "Long Prompt Weighting"
pipeline_lpw = DiffusionPipeline.from_pipe(
    pipeline_original, custom_pipeline="lpw_stable_diffusion"
)

```



---

## Componentes Comunitarios

A veces, no solo el "pipeline" es diferente, sino que el modelo en sí (el UNet, el VAE o el Scheduler) es una implementación personalizada que no existe oficialmente en la librería.

### Pasos para usar componentes personalizados:

1. **Cargar componentes estándar:** Carga el tokenizador o el codificador de texto de forma habitual.
2. **Cargar el componente personalizado:** Importa la clase de Python específica (por ejemplo, un `UNet` modificado) y cárgala con `.from_pretrained()`.
3. **Instanciar el Pipeline:** Pasa todos los componentes (estándar y personalizados) a la clase del pipeline.

---

## Seguridad: `trust_remote_code`

Cuando un pipeline o componente vive en el Hub de Hugging Face y contiene código Python personalizado, debes usar el argumento `trust_remote_code=True` para permitir que se ejecute.

> [!WARNING]
> **Seguridad primero:** Al usar `trust_remote_code=True`, estás ejecutando código de terceros en tu máquina. Te recomendamos:
> * Inspeccionar el código en el repositorio del Hub.
> * Usar un `revision` (hash de commit) específico para evitar que cambios maliciosos futuros te afecten.
> 
> 

```python
pipeline = DiffusionPipeline.from_pretrained(
    "usuario/modelo-comunitario", 
    trust_remote_code=True, 
    revision="1234567" # Commit específico por seguridad
)

```

---

¿Te gustaría que busquemos algún pipeline comunitario específico para una tarea (como por ejemplo, uno para prompts muy largos o para control de video)?