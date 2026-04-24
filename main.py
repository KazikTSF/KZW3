from time import perf_counter


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

def neh(tasks):
  if not tasks:
    return []

  ordered = sorted(range(len(tasks)), key=lambda idx: sum(tasks[idx]), reverse=True)

  res = [ordered[0]]
  for task in ordered[1:]:
    best_pos = 0
    best_cmax = None

    for j in range(len(res) + 1):
      candidate = res[:j] + [task] + res[j:]
      curr = get_cmax(tasks, candidate)

      if best_cmax is None or curr < best_cmax:
        best_cmax = curr
        best_pos = j

    res = res[:best_pos] + [task] + res[best_pos:]

  return res


def qneh(tasks):
  if not tasks:
    return []

  n = len(tasks)
  m = len(tasks[0])

  priorities = [sum(t) for t in tasks]
  order = sorted(range(n), key=priorities.__getitem__, reverse=True)

  pi = [order[0]]

  forward = [[0] * m for _ in range(2)]
  backwards = [[0] * m for _ in range(2)]

  for job in order[1:]:
    k = len(pi)

    if k + 1 > len(forward):
      forward.extend([[0] * m for _ in range(k + 1 - len(forward))])
      backwards.extend([[0] * m for _ in range(k + 1 - len(backwards))])

    job_t = tasks[job]
    job_t_0 = job_t[0]

    for i in range(1, k + 1):
      task = tasks[pi[i - 1]]
      prev = forward[i - 1]
      curr = forward[i]

      curr[0] = prev[0] + task[0]

      for machine in range(1, m):
        curr[machine] = max(prev[machine], curr[machine - 1]) + task[machine]

    for i in range(k - 1, -1, -1):
      task = tasks[pi[i]]
      nxt = backwards[i + 1]
      curr = backwards[i]

      task_m = task[m - 1]
      curr[m - 1] = nxt[m - 1] + task_m

      for machine in range(m - 2, -1, -1):
        curr[machine] = max(nxt[machine], curr[machine + 1]) + task[machine]

    best_pos = 0
    best_cmax = float("inf")

    for pos in range(k + 1):
      forward_pos = forward[pos]
      backwards_pos = backwards[pos]

      time = forward_pos[0] + job_t_0
      cmax_val = time + backwards_pos[0]

      for machine in range(1, m):
        time = max(time, forward_pos[machine]) + job_t[machine]
        cmax_candidate = time + backwards_pos[machine]
        if cmax_candidate > cmax_val:
          cmax_val = cmax_candidate

      if cmax_val < best_cmax:
        best_cmax = cmax_val
        best_pos = pos

    pi.insert(best_pos, job)

  return pi


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

