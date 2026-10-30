# SI3003 · Inteligencia Artificial — Evaluación 2 (Parte II - Aplicada)

**Profesor:** Jorge Iván Padilla Buriticá · Universidad EAFIT  
**Estudiante:** Santiago Vásquez Arboleda (`1000357979`)  
**Grupo:** 01  

---

## 📌 Contenido del Repositorio

- **`Eval2_Vasquez_Santiago.ipynb`**: Notebook principal resuelto y ejecutado secuencialmente con todas sus salidas, gráficas, tablas de métricas y validaciones `assert`.
- **`datos_examen/`**: Conjunto de datos provisto para la evaluación:
  - Resultados de entrenamiento de YOLOv8 (`scratch`, `pretrained`, `freeze`).
  - Predicciones y ground truth para la evaluación de IoU y NMS (`yolo_test_preds.json`, `yolo_test_gt.json`).
  - Corpus institucional ficticio y preguntas de evaluación (`corpus_reglamento.json`, `preguntas_eval.json`).
  - Embeddings oficiales generados (`corpus_embeddings.npy`, `question_embeddings.npy`) y especificación (`embedding_info.json`).
- **Enunciados y Guías Oficiales**: `Eval2_Parte2_enunciado.pdf`, `EVAL_teorica.pdf`, `Guia_reutilizar_notebooks_Parte2.pdf`.

---

## 🚀 Resumen de Implementación

### Sección A — CNN y Detección de Objetos con YOLO (33 pts)
- **A.1:** Implementación de `conv_out` con fórmula de clase validada con asserts frente a `torch.nn.Conv2d`. Red de la Pregunta 1 (Parte I) construida en PyTorch con exactamente 1,562 parámetros entrenables.
- **A.2:** Comparación de curvas de pérdida y métricas (`mAP50`, `mAP50-95`, precision, recall, tiempo) para las 3 estrategias. Análisis de transfer learning para datasets grandes de dominio distante.
- **A.3:** Implementación de `iou` y `nms` greedy con asserts. NMS class-wise sobre el subconjunto del estudiante ($image\_id \bmod 3 == 0$) y evaluación del trade-off precision-recall en $conf \in \{0.25, 0.50\}$.

### Sección B — Transformers y NLP (30 pts)
- **B.1:** Mecanismo de auto-atención causal desde cero en PyTorch sin funciones prediseñadas. Validación estricta con asserts (filas suman 1, máscara triangular superior y paridad con `F.scaled_dot_product_attention`). Reproducción numérica exacta de la Pregunta 4 (Parte I).
- **B.2:** Generación autorregresiva local con `gpt2`. Top-10 tokens, escalamiento de temperatura $T \in \{0.2, 1.0, 2.0\}$ y cálculo de la entropía de Shannon.
- **B.3:** Tokenización comparativa multilingüe y de código con `tiktoken` (`cl100k_base`), inspección de fragmentación UTF-8 en emojis y análisis del sobrecosto en español.

### Sección C — RAG: Compatibilidad, Búsqueda Híbrida e Índices (32 pts)
- **C.0:** Cliente multiproveedor (`OpenAI` SDK) contra NVIDIA NIM (`nvidia/nemotron-3.5-bolt-e4b`). Asserts de compatibilidad de embeddings ($cos \ge 0.99$) y contraejemplo de incompatibilidad entre modelos de igual dimensión ($384$).
- **C.1:** Búsqueda vectorial exacta por producto matricial sin librerías externas. Recall@3 = 0.9167 (11/12) y MRR = 0.8361.
- **C.2:** BM25 con `rank_bm25` y fusión híbrida Reciprocal Rank Fusion (RRF, $k=60$). Comparativa desglosada por tipo de consulta (`sigla` vs `parafrasis`).
- **C.3:** Benchmark con `faiss-cpu` entre búsqueda exacta (`IndexFlatIP`) y aproximada (`IndexHNSWFlat`, $M=16$, $efSearch \in \{8, 64\}$) sobre 20,000 vectores aleatorios.
- **C.4:** Pipeline RAG end-to-end con prompt de fidelidad estricta y auditoría de Faithfulness ($1.000$) con abstención ante preguntas fuera del corpus.

### Sección D — Declaración de IA y Reproducibilidad (5 pts)
- Declaración de herramientas, modelos utilizados, auditoría de fallas y correcciones, y escaneo de claves con 0 tokens filtrados.

---

## 🔒 Auditoría de Seguridad
El repositorio fue auditado mediante escaneo de patrones regex para prevenir la exposición de credenciales (`NVIDIA_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN`). Todas las variables sensibles se gestionan mediante variables de entorno o Colab Secrets.
