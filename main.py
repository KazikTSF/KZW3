from time import perf_counter
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
def _qneh_numba_core(tasks):
  n, m = tasks.shape
  if n == 0:
    return np.empty(0, dtype=np.int64)

  priorities = np.empty(n, dtype=np.int64)
  for i in range(n):
    row_sum = 0
    for machine in range(m):
      row_sum += tasks[i, machine]
    priorities[i] = row_sum

  # Stable sort by descending priority, matching Python's sorted(..., reverse=True).
  order = np.arange(n, dtype=np.int64)
  for i in range(1, n):
    key_idx = order[i]
    key_priority = priorities[key_idx]
    j = i - 1
    while j >= 0 and priorities[order[j]] < key_priority:
      order[j + 1] = order[j]
      j -= 1
    order[j + 1] = key_idx
  pi = np.empty(n, dtype=np.int64)
  pi_len = 1
  pi[0] = order[0]

  forward = np.zeros((n + 1, m), dtype=np.int64)
  backwards = np.zeros((n + 1, m), dtype=np.int64)

  for idx in range(1, n):
    job = order[idx]
    job_t_0 = tasks[job, 0]

    for i in range(1, pi_len + 1):
      task_idx = pi[i - 1]
      forward[i, 0] = forward[i - 1, 0] + tasks[task_idx, 0]

      for machine in range(1, m):
        prev = forward[i - 1, machine]
        left = forward[i, machine - 1]
        if prev > left:
          forward[i, machine] = prev + tasks[task_idx, machine]
        else:
          forward[i, machine] = left + tasks[task_idx, machine]

    for i in range(pi_len - 1, -1, -1):
      task_idx = pi[i]
      backwards[i, m - 1] = backwards[i + 1, m - 1] + tasks[task_idx, m - 1]

      for machine in range(m - 2, -1, -1):
        nxt = backwards[i + 1, machine]
        right = backwards[i, machine + 1]
        if nxt > right:
          backwards[i, machine] = nxt + tasks[task_idx, machine]
        else:
          backwards[i, machine] = right + tasks[task_idx, machine]

    best_pos = 0
    best_cmax = 9223372036854775807

    for pos in range(pi_len + 1):
      time_val = forward[pos, 0] + job_t_0
      cmax_val = time_val + backwards[pos, 0]

      for machine in range(1, m):
        if time_val < forward[pos, machine]:
          time_val = forward[pos, machine] + tasks[job, machine]
        else:
          time_val = time_val + tasks[job, machine]

        cmax_candidate = time_val + backwards[pos, machine]
        if cmax_candidate > cmax_val:
          cmax_val = cmax_candidate

      if cmax_val < best_cmax:
        best_cmax = cmax_val
        best_pos = pos

    for p in range(pi_len, best_pos, -1):
      pi[p] = pi[p - 1]

    pi[best_pos] = job
    pi_len += 1

  return pi[:pi_len]


def qneh(tasks):
  if not tasks:
    return []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  if tasks_np.ndim != 2:
    raise ValueError("tasks must be a 2D array-like structure")
  return _qneh_numba_core(tasks_np).tolist()



if __name__ == "__main__":
  total_qneh_time = 0.0

  for i in range(121):
    tasksInput = get_tasks(f"data/data{i}.txt")
    print("=" * 20)
    print("Calculating data " + str(i))

    start = perf_counter()
    qneh_order = qneh(tasksInput)
    elapsed = perf_counter() - start
    total_qneh_time += elapsed

    print(qneh_order)
    print(get_cmax(tasksInput, qneh_order))
    print(f"qneh time: {elapsed:.6f} s")

  print("=" * 20)
  print(f"Total qneh time: {total_qneh_time:.6f} s")

