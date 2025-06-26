# Clasificación de Imágenes con Redes Residuales (ResNet)

Este repositorio contiene la implementación desde cero de una red neuronal residual (ResNet) para la clasificación de imágenes del conjunto de datos **Fruits-360** (resolución 100×100). Se parte del fundamento teórico de las ResNets y se aplica dicho conocimiento en un caso práctico utilizando PyTorch.

---

## 1. Fundamento Teórico: Redes Residuales

Las **ResNets** (He et al., 2015) surgieron como respuesta al problema de **degradación en redes profundas**: al aumentar la profundidad de una red, la precisión de entrenamiento puede empeorar, incluso sin sobreajuste. Esto sugiere dificultades de optimización.

### 1.1. Motivación

Dado un mapeo objetivo $H(x)$, una red tradicional intenta aproximar directamente:
$$
\mathcal{F}(x) \approx H(x)
$$

ResNet reformula esta tarea como el aprendizaje de un **residuo**:
$$
\mathcal{F}(x) := H(x) - x \quad \Rightarrow \quad H(x) = \mathcal{F}(x) + x
$$

Se agrega así una **conexión residual o atajo** (`shortcut`) que permite que el flujo de gradientes durante backpropagation atraviese más fácilmente la red, estabilizando el entrenamiento.

### 1.2. Estructura del Bloque Residual

Un bloque residual con dos capas puede representarse como:

```text
Input
  │
[Conv → BN → ReLU → Conv → BN]
  │            ▲
  └── Shortcut ┘
      (suma)
       │
     ReLU


