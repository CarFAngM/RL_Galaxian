# RL Galaxian - Proyecto DQN

Estructura propuesta para entrenar un agente DQN en el entorno `ALE/Galaxian-v5`.

Archivos principales:

- `src/dqn.py`: definición de la red convolucional DQN.
- `src/replay.py`: replay buffer.
- `src/agent.py`: agente DQN (select_action, train_step, save/load).
- `src/utils.py`: preprocesamiento de frames, stacking y funciones de plotting.
- `src/train.py`: función `train_agent` con early stopping y checkpoints.
- `src/record.py`: grabación de episodios en video.
- `train.py`: script CLI para entrenar.
- `record_cli.py`: script CLI para grabar usando un modelo guardado.
- `requirements.txt`: dependencias.

Quick start (Windows PowerShell):

```powershell
pip install -r requirements.txt
python train.py --episodes 100 --email estudiante@uvg.edu.gt
python record_cli.py --model checkpoints\best_model_estudiante.pth --email estudiante@uvg.edu.gt
```

Notas:
- El preprocesamiento convierte frames a escala de grises 84x84 y apila 4 frames.
- Early stopping: basado en media móvil de recompensas (configurable).
- Los modelos se guardan en `checkpoints/`.
