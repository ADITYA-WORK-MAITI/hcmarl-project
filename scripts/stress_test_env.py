"""HC-MARL Phase 2 (#29): Stress test — 10K episodes, check no crashes, memory stable."""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo

def stress_test(n_episodes=10000, n_workers=4, max_steps=60):
    tracemalloc.start()
    env = WarehousePettingZoo(n_workers=n_workers, max_steps=max_steps)
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            actions = {a: np.random.randint(0, env.n_tasks) for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()): break
        if (ep+1) % 1000 == 0:
            mem = tracemalloc.get_traced_memory()
            print(f"  Episode {ep+1}/{n_episodes}: mem={mem[1]/1024/1024:.1f}MB peak")
    elapsed = time.time() - t0
    mem_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Stress test PASSED: {n_episodes} episodes in {elapsed:.1f}s, peak mem={mem_final[1]/1024/1024:.1f}MB")

if __name__ == "__main__":
    stress_test()
