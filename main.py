from time import perf_counter
import csv
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
  if len(tasks) == 0:
    return []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  if tasks_np.ndim != 2:
    raise ValueError("tasks must be a 2D array-like structure")
  return _qneh_numba_core(tasks_np).tolist()

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
def _neh_numba_core(tasks):
  n, m = tasks.shape
  if n == 0:
    return np.empty(0, dtype=np.int64)

  priorities = np.empty(n, dtype=np.int64)
  for i in range(n):
    row_sum = 0
    for machine in range(m):
      row_sum += tasks[i, machine]
    priorities[i] = row_sum

  order = np.arange(n, dtype=np.int64)
  for i in range(1, n):
    key_idx = order[i]
    key_priority = priorities[key_idx]
    j = i - 1
    while j >= 0 and priorities[order[j]] < key_priority:
      order[j + 1] = order[j]
      j -= 1
    order[j + 1] = key_idx

  res = np.empty(n, dtype=np.int64)
  candidate = np.empty(n, dtype=np.int64)
  res[0] = order[0]
  res_len = 1

  for idx in range(1, n):
    job = order[idx]
    best_pos = 0
    best_cmax = 9223372036854775807

    for pos in range(res_len + 1):
      for p in range(pos):
        candidate[p] = res[p]
      candidate[pos] = job
      for p in range(pos, res_len):
        candidate[p + 1] = res[p]

      curr = _get_cmax_numba(tasks, candidate, res_len + 1)
      if curr < best_cmax:
        best_cmax = curr
        best_pos = pos

    for p in range(res_len, best_pos, -1):
      res[p] = res[p - 1]
    res[best_pos] = job
    res_len += 1

  return res[:res_len]


def neh(tasks):
  if len(tasks) == 0:
    return []

  tasks_np = np.asarray(tasks, dtype=np.int64)
  if tasks_np.ndim != 2:
    raise ValueError("tasks must be a 2D array-like structure")
  return _neh_numba_core(tasks_np).tolist()


if __name__ == "__main__":
  total_qneh_time = 0.0
  total_neh_time = 0.0

  csv_path = "benchmark_results.csv"
  with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
      "input_id",
      "filename",
      "n",
      "machines",
      "qneh_time_s",
      "neh_time_s",
      "qneh_cmax",
      "neh_cmax",
      "qneh_order",
      "neh_order",
    ])

    for i in range(121):
      filename = f"data/data{i}.txt"
      tasksInput = get_tasks(filename)
      n = len(tasksInput)
      machines = len(tasksInput[0]) if tasksInput else 0

      print("=" * 20)
      print("Calculating data " + str(i))

      start = perf_counter()
      qneh_order = qneh(tasksInput)
      qneh_elapsed = perf_counter() - start
      total_qneh_time += qneh_elapsed
      qneh_cmax = get_cmax(tasksInput, qneh_order)

      start = perf_counter()
      neh_order = neh(tasksInput)
      neh_elapsed = perf_counter() - start
      total_neh_time += neh_elapsed
      neh_cmax = get_cmax(tasksInput, neh_order)

      print(f"qneh order: {qneh_order}")
      print(f"qneh cmax: {qneh_cmax}")
      print(f"qneh time: {qneh_elapsed:.6f} s")
      print(f"neh order: {neh_order}")
      print(f"neh cmax: {neh_cmax}")
      print(f"neh time: {neh_elapsed:.6f} s")

      writer.writerow([
        i,
        filename,
        n,
        machines,
        f"{qneh_elapsed:.9f}",
        f"{neh_elapsed:.9f}",
        qneh_cmax,
        neh_cmax,
        " ".join(map(str, qneh_order)),
        " ".join(map(str, neh_order)),
      ])

  print("=" * 20)
  print(f"Total qneh time: {total_qneh_time:.6f} s")
  print(f"Total neh time: {total_neh_time:.6f} s")
  print(f"CSV saved to: {csv_path}")

