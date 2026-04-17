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

  return currentEnd[-1]

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

if __name__ == "__main__":
  for i in range(121):
    tasksInput = get_tasks(f"data/data{i}.txt")
    print("="*20)
    print("Calculating data " + str(i))
    print(neh(tasksInput))
    print(get_cmax(tasksInput, neh(tasksInput)))
