
---

# Diffusers

Diffusers es una biblioteca de modelos de difusión preentrenados de última generación para generar videos, imágenes y audio.

La biblioteca gira en torno a [DiffusionPipeline](/docs/diffusers/v0.36.0/en/api/pipelines/overview#diffusers.DiffusionPipeline), una API diseñada para:

- realizar inferencias de forma sencilla con solo unas pocas líneas de código
- tener la flexibilidad de combinar componentes de pipelines (modelos, planificadores)
- cargar y usar adaptadores como LoRA

Diffusers también incluye optimizaciones —como la descarga y la cuantificación— para garantizar que incluso los modelos más grandes sean accesibles en dispositivos con poca memoria. Si la memoria no es un problema, Diffusers admite torch.compile para aumentar la velocidad de inferencia.

¡Empieza ya mismo con un modelo de Diffusers en el [Hub](https://huggingface.co/models?library=diffusers&sort=trending)!

## Aprende

Si eres principiante, te recomendamos empezar con el [Curso de Modelos de Difusión de Hugging Face](https://huggingface.co/learn/diffusion-course/unit0/1). Aprenderás la teoría detrás de los modelos de difusión y cómo usar la biblioteca Diffusers para generar imágenes, afinar tus propios modelos y mucho más.

---
