# 🎮 RL Galaxian - Deep Reinforcement Learning

Proyecto completo de **Reinforcement Learning** para entrenar agentes que jueguen Galaxian usando **Double DQN**, **Actor-Critic (A2C)** y **PPO**.

## 🚀 Características

- ✅ **Double DQN**: Reduce sobreestimación de Q-values usando dos redes (policy y target)
- ✅ **Actor-Critic (A2C)**: Policy gradient con función de valor
- ✅ **PPO (Proximal Policy Optimization)**: Algoritmo state-of-the-art con clipping de ratio
- ✅ **Optimización por reward**: Guarda mejor modelo basado en promedio móvil de recompensas
- ✅ **Memory-optimized**: Replay buffer con almacenamiento uint8 (4x menos memoria)
- ✅ **Reentrenamiento**: Carga modelo + buffer para continuar entrenando
- ✅ **Jupyter Notebooks**: Interfaz interactiva para entrenamiento y análisis
- ✅ **Video recording**: Graba episodios del agente entrenado

## 📁 Estructura del Proyecto

```
RL_Galaxian/
├── src/
│   ├── dqn.py              # Red neuronal Double DQN
│   ├── agent.py            # Agente DQN con train_step y save/load
│   ├── replay.py           # Replay buffer optimizado (uint8)
│   ├── actor_critic.py     # Red Actor-Critic compartida
│   ├── ac_agent.py         # Agente A2C
│   ├── train.py            # Función de entrenamiento DQN
│   ├── train_ac.py         # Función de entrenamiento A2C
│   ├── record.py           # Grabación de videos
│   └── utils.py            # Preprocesamiento y utilidades
├── train_dqn.ipynb         # Notebook: Entrenamiento Double DQN
├── train_ac_notebook.ipynb # Notebook: Entrenamiento Actor-Critic
├── PPO_RL.ipynb            # Notebook: Entrenamiento PPO
├── train.py                # CLI: Entrenar DQN
├── train_ac_cli.py         # CLI: Entrenar Actor-Critic
├── record_cli.py           # CLI: Grabar videos
└── requirements.txt        # Dependencias
```

## 🛠️ Instalación

```powershell
# Clonar repositorio
git clone <repo-url>
cd RL_Galaxian

# Crear entorno virtual (recomendado)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Uso Rápido

### Opción 1: Jupyter Notebooks (Recomendado)

```powershell
# Abrir notebooks
jupyter notebook

# Ejecutar:
# - train_dqn.ipynb: Para Double DQN
# - train_ac_notebook.ipynb: Para Actor-Critic
# - PPO_RL.ipynb: Para PPO
```

### Opción 2: Scripts CLI

**Entrenar Double DQN:**
```powershell
python train.py --episodes 500 --email tu@email.com
```

**Entrenar Actor-Critic:**
```powershell
python train_ac_cli.py --episodes 500 --email tu@email.com
```

**Grabar video del agente:**
```powershell
python record_cli.py --model checkpoints_dqn\best_model_tu.pth --email tu@email.com
```

## 🧠 Algoritmos Implementados

### Double DQN
- **Arquitectura**: 4 capas convolucionales + 4 capas fully-connected
- **Optimización**: Adam, lr=1e-4
- **Exploración**: ε-greedy con decay exponencial
- **Buffer**: 100K experiencias (uint8 para eficiencia)
- **Target update**: Cada 1000 steps
- **Early stopping**: Basado en moving average de rewards

### Actor-Critic (A2C)
- **Arquitectura**: Capas conv compartidas, heads separados (actor/critic)
- **Optimización**: Adam, lr=3e-4
- **Entropy regularization**: 0.05 (fomenta exploración)
- **Advantages**: Normalizadas para estabilidad
- **Sin reward clipping**: Aprende valores reales de Galaxian
- **Early stopping**: Basado en moving average de rewards

### PPO (Proximal Policy Optimization)
- **Arquitectura**: Red compartida con heads actor/critic separados
- **Optimización**: Adam, lr=3e-4
- **Clipped objective**: Ratio clipping (ε=0.2) para actualizaciones estables
- **Multiple epochs**: 4 épocas de actualización por batch
- **GAE (Generalized Advantage Estimation)**: λ=0.95 para reducir varianza
- **Entropy bonus**: 0.01 para exploración
- **Value function clipping**: Estabiliza aprendizaje del crítico
- **State-of-the-art**: Mejor balance exploración/explotación

## 📊 Hiperparámetros Clave

| Parámetro | Double DQN | Actor-Critic | PPO |
|-----------|------------|--------------|-----|
| Learning Rate | 1e-4 | 3e-4 | 3e-4 |
| Batch Size | 32 | N/A (on-policy) | 256 |
| Gamma (γ) | 0.99 | 0.99 | 0.99 |
| Epsilon start | 1.0 | N/A | N/A |
| Epsilon end | 0.10 | N/A | N/A |
| Clip range (ε) | N/A | N/A | 0.2 |
| Entropy coef | N/A | 0.05 | 0.01 |
| GAE λ | N/A | N/A | 0.95 |
| Update epochs | N/A | 1 | 4 |
| MA Window | 20 | 20 | 20 |

## 📈 Métricas y Visualización

Los notebooks generan automáticamente:
- Gráficas de rewards por episodio
- TD Loss / Actor-Critic losses / PPO losses
- Epsilon decay (DQN) / Entropy (A2C/PPO)
- Policy ratio y clipping (PPO)
- Moving average de rewards
- Gráficas guardadas en `checkpoints_*/`

## 💾 Checkpoints

Los modelos se guardan en:
- `checkpoints_dqn/`: Modelos Double DQN
- `checkpoints_ac/`: Modelos Actor-Critic
- `checkpoints_ppo/`: Modelos PPO

Tipos de checkpoints:
- `best_model_*.pth`: Mejor modelo (mayor MA de rewards)
- `final_model_*.pth`: Modelo al finalizar entrenamiento
- `checkpoint_*_ep{N}_*.pth`: Checkpoints periódicos

## 🎬 Videos

Los videos se guardan en:
- `videos_dqn/`: Videos de agente DQN
- `videos_ac/`: Videos de agente Actor-Critic
- `videos_ppo/`: Videos de agente PPO

Formato: MP4 con metadata del episodio

## 🔧 Preprocesamiento

1. Conversión a escala de grises
2. Resize a 84x84
3. Normalización [0, 1]
4. Frame stacking (4 frames)

## 📝 Notas Importantes

- **Rewards no clipeados en A2C**: Aprende valores reales de Galaxian (+30, +60, +200)
- **DQN usa rewards clipeados**: [-1, +1] para estabilidad
- **Replay buffer en uint8**: Ahorra 75% de memoria
- **Early stopping automático**: Detiene si no mejora en 200 episodios
- **Reentrenamiento**: Soporta carga de modelo + buffer para continuar

## 🐛 Troubleshooting

**Error de memoria:**
- Reduce `BUFFER_SIZE` o `EPISODES`
- El buffer ya está optimizado con uint8

**Modelo no aprende:**
- Verifica que `MA_WINDOW` sea apropiado
- Aumenta `EPISODES` para más exploración
- Revisa gráficas de entropy/epsilon para asegurar exploración

**Error de checkpoint:**
- Asegúrate de usar PyTorch 2.6+
- Los checkpoints incluyen `weights_only=True` para seguridad

## 🎓 Autor

Proyecto de Reinforcement Learning - UVG
Email: ang23010@uvg.edu.gt

## 📜 Licencia

MIT License - Libre para uso académico y personal

Readme generado con IA
