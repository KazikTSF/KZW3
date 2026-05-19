from time import perf_counter
import csv
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    total_sa_time = 0.0

    sa_history_path = "sa_iterations.csv"
    with open(sa_history_path, "w", newline="", encoding="utf-8") as sa_history_file:
        history_writer = csv.writer(sa_history_file)
        history_writer.writerow(["iteration", "cmax"])

        input_file = "data/data80.txt"

        tasksInput = get_tasks(input_file)
        n = len(tasksInput)
        machines = len(tasksInput[0]) if tasksInput else 0

        print("=" * 20)
        print("Calculating data for " + input_file)

        start = perf_counter()
        sa_order, sa_history = simulated(tasksInput, list(range(n)))
        sa_elapsed = perf_counter() - start
        total_sa_time += sa_elapsed
        sa_cmax = get_cmax(tasksInput, sa_order)

        print(f"sa order: {sa_order}")
        print(f"sa cmax: {sa_cmax}")
        print(f"sa time: {sa_elapsed:.6f} s")

        iterations_plot = []
        cmax_plot = []

        for i, cmax in enumerate(sa_history):
            if i % 200 == 0:
                history_writer.writerow([i, cmax])
                iterations_plot.append(i)
                cmax_plot.append(cmax)

    # Generate chart
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_plot, cmax_plot, label="Simulated Annealing CMAX", color="blue")
    plt.title("SA Iterations vs CMAX")
    plt.xlabel("Iteration")
    plt.ylabel("CMAX")
    plt.grid(True)
    plt.legend()
    chart_path = "sa_iterations_chart.png"
    plt.savefig(chart_path)
    print(f"SA iterations chart saved to: {chart_path}")

    print("=" * 20)
    print(f"Total sa time: {total_sa_time:.6f} s")
    print(f"SA iterations CSV saved to: {sa_history_path}")