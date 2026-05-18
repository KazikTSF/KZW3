import csv
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def get_tasks(path):
  tasks = []
  with open(path, "r", encoding="utf-8") as file:
    header = file.readline().split()
    if len(header) != 2:
      raise Exception("Invalid input file format")

    n, machines = map(int, header)

    for _ in range(n):
      timeList = file.readline().split()
      if len(timeList) != machines:
        raise Exception("Invalid input file format")
      tasks.append([int(x) for x in timeList])

  return tasks


def get_cmax(tasks, order):
  if not order:
    return 0

  machines = len(tasks[0])
  currentEnd = [0] * machines

  first_task = tasks[order[0]]
  currentEnd[0] = first_task[0]
  for m in range(1, machines):
    currentEnd[m] = currentEnd[m - 1] + first_task[m]

  for i in range(1, len(order)):
    task = tasks[order[i]]
    currentEnd[0] += task[0]
    for m in range(1, machines):
      currentEnd[m] = max(currentEnd[m], currentEnd[m - 1]) + task[m]

  return int(currentEnd[-1])


@njit(cache=True)
def _get_cmax_numba(tasks, order, order_len):
  if order_len == 0:
    return 0

  machines = tasks.shape[1]
  currentEnd = np.zeros(machines, dtype=np.int64)

  first_task = order[0]
  currentEnd[0] = tasks[first_task, 0]
  for m in range(1, machines):
    currentEnd[m] = currentEnd[m - 1] + tasks[first_task, m]

  for i in range(1, order_len):
    task_idx = order[i]
    currentEnd[0] += tasks[task_idx, 0]
    for m in range(1, machines):
      prev = currentEnd[m]
      left = currentEnd[m - 1]
      if prev > left:
        currentEnd[m] = prev + tasks[task_idx, m]
      else:
        currentEnd[m] = left + tasks[task_idx, m]

  return int(currentEnd[-1])


@njit(cache=True)
def _sa_numba_core(tasks, initial_order):
  n = tasks.shape[0]
  if n == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

  current_order = initial_order.copy()
  current_cmax = _get_cmax_numba(tasks, current_order, n)

  best_order = current_order.copy()
  best_cmax = current_cmax

  temp = 5000.0
  temp_mod = 0.9999

  history = np.empty(100000, dtype=np.int64)

  for i in range(100001):
    rand1 = np.random.randint(0, n)
    rand2 = np.random.randint(0, n)

    new_order = current_order.copy()
    val = new_order[rand1]

    if rand1 < rand2:
      new_order[rand1:rand2] = new_order[rand1 + 1:rand2 + 1]
    elif rand1 > rand2:
      new_order[rand2 + 1:rand1 + 1] = new_order[rand2:rand1]

    new_order[rand2] = val

    new_cmax = _get_cmax_numba(tasks, new_order, n)

    delta = new_cmax - current_cmax

    if delta < 0 or np.random.random() < np.exp(-delta / temp):
      current_order = new_order
      current_cmax = new_cmax

      if current_cmax < best_cmax:
        best_cmax = current_cmax
        best_order = current_order.copy()

    temp *= temp_mod
    history[i] = current_cmax

  return best_order, history


def ts(tasks, initial_order=None):
  if len(tasks) == 0:
    return []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  n = tasks_np.shape[0]

  if initial_order is None:
    initial_order_np = np.arange(n, dtype=np.int64)
  else:
    initial_order_np = np.asarray(initial_order, dtype=np.int64)

  best_order, _ = _ts_numba_core(tasks_np, initial_order_np)
  return best_order.tolist()

@njit(cache=True)
def _ts_find_best_move(tasks, order):
  n = order.shape[0]
  best_move = np.array([0, 1], dtype=np.int64)
  best_cmax = _get_cmax_numba(tasks, order, n)
  
  for i in range(n):
    for j in range(n):
      if i == j:
        continue
      new_order = order.copy()
      element = new_order[i]
      
      if i < j:
        new_order[i:j] = new_order[i + 1:j + 1]
      else:
        new_order[j + 1:i + 1] = new_order[j:i]
      
      new_order[j] = element
      new_cmax = _get_cmax_numba(tasks, new_order, n)
      
      if new_cmax < best_cmax:
        best_cmax = new_cmax
        best_move = np.array([i, j], dtype=np.int64)
  
  return best_move

@njit(cache=True)
def _ts_add_move_tabu(move, order, tabu, tabu_size):
  if tabu.shape[0] > 0:
    if tabu.shape[0] >= tabu_size:
      tabu[:-1] = tabu[1:]
    
    job = order[move[1]]
    tabu[-1, 0] = job
    tabu[-1, 1] = move[1]


@njit(cache=True)
def _ts_step(tasks, order, tabu, tabu_size):
  move = _ts_find_best_move(tasks, order)
  _ts_add_move_tabu(move, order, tabu, tabu_size)

  i = move[0]
  j = move[1]
  element = order[i]
  
  if i < j:
    order[i:j] = order[i + 1:j + 1]
  else:
    order[j + 1:i + 1] = order[j:i]
  
  order[j] = element
  return order


@njit(cache=True)
def _ts_numba_core(tasks, initial_order):
  n = tasks.shape[0]
  if n == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
  
  tabu_size = 10

  current_order = initial_order.copy()
  current_cmax = _get_cmax_numba(tasks, current_order, n)
  
  best_order = current_order.copy()
  best_cmax = current_cmax

  tabu = np.zeros((tabu_size, 2), dtype=np.int64)

  history = np.empty(101, dtype=np.int64)
  history[0] = current_cmax
  
  for i in range(101):
    _ts_step(tasks, current_order, tabu, tabu_size)
    current_cmax = _get_cmax_numba(tasks, current_order, n)
    history[i] = current_cmax
    
    if current_cmax < best_cmax:
      best_cmax = current_cmax
      best_order = current_order.copy()
  
  return best_order, history
  


def simulated(tasks, initial_order=None):
  if len(tasks) == 0:
    return [], []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  n = tasks_np.shape[0]

  if initial_order is None:
    initial_order_np = np.arange(n, dtype=np.int64)
  else:
    initial_order_np = np.asarray(initial_order, dtype=np.int64)

  best_order, history = _sa_numba_core(tasks_np, initial_order_np)
  return best_order.tolist(), history.tolist()


def tabu(tasks, initial_order=None):
  if len(tasks) == 0:
    return [], []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  n = tasks_np.shape[0]

  if initial_order is None:
    initial_order_np = np.arange(n, dtype=np.int64)
  else:
    initial_order_np = np.asarray(initial_order, dtype=np.int64)

  best_order, history = _ts_numba_core(tasks_np, initial_order_np)
  return best_order.tolist(), history.tolist()


if __name__ == "__main__":
  total_sa_time = 0.0
  total_ts_time = 0.0

  sa_history_path = "sa_iterations.csv"
  ts_history_path = "ts_iterations.csv"

  input_file = "data/data80.txt"
  tasksInput = get_tasks(input_file)
  n = len(tasksInput)
  machines = len(tasksInput[0]) if tasksInput else 0

  print("=" * 20)
  print("Calculating data for " + input_file)

  # Simulated Annealing
  with open(sa_history_path, "w", newline="",
            encoding="utf-8") as sa_history_file:
    history_writer = csv.writer(sa_history_file)
    history_writer.writerow(["iteration", "cmax"])

    start = perf_counter()
    sa_order, sa_history = simulated(tasksInput, list(range(n)))
    sa_elapsed = perf_counter() - start
    total_sa_time += sa_elapsed
    sa_cmax = get_cmax(tasksInput, sa_order)

    print("\n--- Simulated Annealing ---")
    print(f"sa order: {sa_order}")
    print(f"sa cmax: {sa_cmax}")
    print(f"sa time: {sa_elapsed:.6f} s")

    sa_iterations_plot = []
    sa_cmax_plot = []

    for i, cmax in enumerate(sa_history):
      if i % 200 == 0:
        history_writer.writerow([i, cmax])
        sa_iterations_plot.append(i)
        sa_cmax_plot.append(cmax)

  # Tabu Search
  with open(ts_history_path, "w", newline="",
            encoding="utf-8") as ts_history_file:
    history_writer = csv.writer(ts_history_file)
    history_writer.writerow(["iteration", "cmax"])

    start = perf_counter()
    ts_order, ts_history = tabu(tasksInput, list(range(n)))
    ts_elapsed = perf_counter() - start
    total_ts_time += ts_elapsed
    ts_cmax = get_cmax(tasksInput, ts_order)

    print("\n--- Tabu Search ---")
    print(f"ts order: {ts_order}")
    print(f"ts cmax: {ts_cmax}")
    print(f"ts time: {ts_elapsed:.6f} s")

    ts_iterations_plot = []
    ts_cmax_plot = []

    for i, cmax in enumerate(ts_history):
      history_writer.writerow([i, cmax])
      ts_iterations_plot.append(i)
      ts_cmax_plot.append(cmax)

  plt.figure(figsize=(12, 6))
  plt.plot(ts_iterations_plot, ts_cmax_plot, label="Tabu Search CMAX",
           color="red", marker='s', markersize=3)
  plt.title("TS Iterations vs CMAX")
  plt.xlabel("Iteration")
  plt.ylabel("CMAX")
  plt.grid(True)
  plt.legend()
  chart_path = "ts_iterations_chart.png"
  plt.savefig(chart_path)
  print(f"\nComparison chart saved to: {chart_path}")

  print("=" * 20)
  print(f"Total SA time: {total_sa_time:.6f} s")
  print(f"Total TS time: {total_ts_time:.6f} s")
  print(f"SA iterations CSV saved to: {sa_history_path}")
  print(f"TS iterations CSV saved to: {ts_history_path}")
