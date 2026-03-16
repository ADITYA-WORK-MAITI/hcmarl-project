"""Create and execute notebooks/env_smoke_test.ipynb programmatically."""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
nb.metadata['kernelspec'] = {
    'name': 'hcmarl',
    'display_name': 'HC-MARL',
    'language': 'python'
}

# Cell 1: Imports and setup
nb.cells.append(new_markdown_cell("# Environment Smoke Test\nRun 1 episode (60 steps) with 4 workers, random actions. Verify 3CC-r conservation law and plot fatigue dynamics."))

nb.cells.append(new_code_cell("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('.'))))
sys.path.insert(0, r'C:\\Users\\admin\\Desktop\\hcmarl_project')
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo

env = WarehousePettingZoo(n_workers=4, max_steps=60)
obs, infos = env.reset(seed=42)
print(f"Environment created: {env.n_workers} workers, {env.n_muscles} muscles, {env.n_tasks} tasks")
print(f"Muscle groups: {env.muscle_names}")
print(f"Tasks: {env.task_names}")
print(f"Observation dim: {env.obs_dim}")
"""))

# Cell 2: Run episode and collect data
nb.cells.append(new_code_cell("""# Run 1 episode (60 steps) with random actions
n_steps = 60
worker_idx = 0  # Track worker_0 for plotting
muscle_names = env.muscle_names

# Storage: [steps, muscles, 3] for MR/MA/MF
history = {m: {'MR': [], 'MA': [], 'MF': []} for m in muscle_names}
conservation_errors = []

obs, _ = env.reset(seed=42)
for step in range(n_steps):
    actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Record states for worker_0
    for m in muscle_names:
        s = env.states[worker_idx][m]
        history[m]['MR'].append(s['MR'])
        history[m]['MA'].append(s['MA'])
        history[m]['MF'].append(s['MF'])

    # Conservation check for ALL workers, ALL muscles
    for wi in range(env.n_workers):
        for m in muscle_names:
            s = env.states[wi][m]
            total = s['MR'] + s['MA'] + s['MF']
            conservation_errors.append(abs(total - 1.0))

print(f"Episode complete: {n_steps} steps")
print(f"Max conservation error: {max(conservation_errors):.2e}")
print(f"Mean conservation error: {np.mean(conservation_errors):.2e}")
assert max(conservation_errors) < 1e-5, f"Conservation law violated! Max error: {max(conservation_errors)}"
print("Conservation law MR+MA+MF=1 VERIFIED at every step for all workers and muscles (tol=1e-5).")
"""))

# Cell 3: Plot MR/MA/MF for shoulder
nb.cells.append(new_code_cell("""# Plot 1: MR/MA/MF trajectories for shoulder muscle (worker_0)
fig, ax = plt.subplots(figsize=(10, 5))
steps = np.arange(1, n_steps + 1)
muscle = 'shoulder'

ax.plot(steps, history[muscle]['MR'], 'b-', linewidth=2, label='MR (Resting)')
ax.plot(steps, history[muscle]['MA'], 'r-', linewidth=2, label='MA (Active)')
ax.plot(steps, history[muscle]['MF'], 'g-', linewidth=2, label='MF (Fatigued)')
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('Compartment Value', fontsize=12)
ax.set_title(f'3CC-r Fatigue Dynamics - {muscle.title()} Muscle (Worker 0)', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(r'C:\\Users\\admin\\Desktop\\hcmarl_project\\notebooks\\smoke_fig1_shoulder_dynamics.png', dpi=150)
plt.show()
print("Saved: notebooks/smoke_fig1_shoulder_dynamics.png")
"""))

# Cell 4: Fatigue heatmap
nb.cells.append(new_code_cell("""# Plot 2: Fatigue heatmap across all 6 muscles over time (worker_0)
fatigue_matrix = np.array([history[m]['MF'] for m in muscle_names])  # shape: (6, 60)

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(fatigue_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
               extent=[1, n_steps, len(muscle_names)-0.5, -0.5])
ax.set_yticks(range(len(muscle_names)))
ax.set_yticklabels([m.title() for m in muscle_names], fontsize=11)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_title('Fatigue (MF) Heatmap - All Muscles (Worker 0)', fontsize=14)
cbar = fig.colorbar(im, ax=ax, label='MF (Fatigue Level)')
fig.tight_layout()
fig.savefig(r'C:\\Users\\admin\\Desktop\\hcmarl_project\\notebooks\\smoke_fig2_fatigue_heatmap.png', dpi=150)
plt.show()
print("Saved: notebooks/smoke_fig2_fatigue_heatmap.png")
"""))

# Cell 5: Summary
nb.cells.append(new_code_cell("""# Summary
print("="*60)
print("ENVIRONMENT SMOKE TEST - RESULTS")
print("="*60)
print(f"Workers:           {env.n_workers}")
print(f"Steps completed:   {n_steps}")
print(f"Muscles:           {len(muscle_names)} ({', '.join(muscle_names)})")
print(f"Tasks:             {env.n_tasks} ({', '.join(env.task_names)})")
print(f"Conservation law:  PASSED (max error {max(conservation_errors):.2e})")
print(f"Figures saved:     2 (smoke_fig1, smoke_fig2)")
print("="*60)
print("ALL CHECKS PASSED")
"""))

# Write notebook
nb_path = r'C:\Users\admin\Desktop\hcmarl_project\notebooks\env_smoke_test.ipynb'
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)
print(f"Notebook written to {nb_path}")
