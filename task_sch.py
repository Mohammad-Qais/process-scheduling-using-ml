import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate synthetic dataset
def generate_tasks(n=100, seed=42):
    random.seed(seed)
    tasks = []
    for i in range(n):
        arrival = random.randint(0, 50)
        burst = random.randint(1, 20)
        tasks.append((i, arrival, burst))
    return pd.DataFrame(tasks, columns=["TaskID", "ArrivalTime", "BurstTime"])

# Step 2: Apply SJF Scheduling
def sjf_scheduling(df):
    tasks = df.sort_values(by=["ArrivalTime", "BurstTime"]).copy()
    time = 0
    waiting_times = []
    turnaround_times = []
    scheduled_order = []
    ready_queue = []
    
    while not tasks.empty or ready_queue:
        arrived = tasks[tasks['ArrivalTime'] <= time]
        if not arrived.empty:
            next_task = arrived.loc[arrived['BurstTime'].idxmin()]
            ready_queue.append(next_task)
            tasks = tasks.drop(next_task.name)
        
        if ready_queue:
            current = min(ready_queue, key=lambda x: x['BurstTime'])
            ready_queue.remove(current)
            start_time = max(time, current['ArrivalTime'])
            wait_time = start_time - current['ArrivalTime']
            turn_time = wait_time + current['BurstTime']
            waiting_times.append(wait_time)
            turnaround_times.append(turn_time)
            scheduled_order.append(current['TaskID'])
            time = start_time + current['BurstTime']
        else:
            time += 1
    
    return scheduled_order, np.mean(waiting_times), np.mean(turnaround_times)

def fcfs_scheduling(df):
    tasks = df.sort_values(by=["ArrivalTime"]).copy()
    time = 0
    waiting_times = []
    turnaround_times = []
    scheduled_order = []

    for _, task in tasks.iterrows():
        start_time = max(time, task['ArrivalTime'])
        wait_time = start_time - task['ArrivalTime']
        turn_time = wait_time + task['BurstTime']
        waiting_times.append(wait_time)
        turnaround_times.append(turn_time)
        scheduled_order.append(task['TaskID'])
        time = start_time + task['BurstTime']

    return scheduled_order, np.mean(waiting_times), np.mean(turnaround_times)


# Step 3: Train ML model to predict shortest job from ready queue
def prepare_training_data(df):
    df_sorted = df.sort_values(by="ArrivalTime").reset_index(drop=True)
    features = []
    labels = []
    for i in range(len(df_sorted) - 2):
        task_set = df_sorted.iloc[i:i+3]
        if len(task_set) < 2:
            continue
        shortest_index = task_set['BurstTime'].idxmin() - i
        for j in range(len(task_set)):
            features.append([task_set.iloc[j]['ArrivalTime'], task_set.iloc[j]['BurstTime']])
            labels.append(1 if j == shortest_index else 0)
    return np.array(features), np.array(labels)

def ml_model_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# Step 4: ML-based scheduling simulator
def ml_scheduler(df, model):
    tasks = df.copy()
    time = 0
    waiting_times = []
    turnaround_times = []
    scheduled_order = []
    ready_queue = []

    while not tasks.empty or ready_queue:
        arrived = tasks[tasks['ArrivalTime'] <= time]
        for _, row in arrived.iterrows():
            ready_queue.append(row)
            tasks = tasks.drop(row.name)

        if ready_queue:
            if len(ready_queue) == 1:
                chosen = ready_queue.pop(0)
            else:
                feature_set = [[task['ArrivalTime'], task['BurstTime']] for task in ready_queue]
                predictions = model.predict(feature_set)
                try:
                    chosen_idx = predictions.tolist().index(1)
                except ValueError:
                    chosen_idx = np.argmin([task['BurstTime'] for task in ready_queue])
                chosen = ready_queue.pop(chosen_idx)

            start_time = max(time, chosen['ArrivalTime'])
            wait_time = start_time - chosen['ArrivalTime']
            turn_time = wait_time + chosen['BurstTime']
            waiting_times.append(wait_time)
            turnaround_times.append(turn_time)
            scheduled_order.append(chosen['TaskID'])
            time = start_time + chosen['BurstTime']
        else:
            time += 1

    return scheduled_order, np.mean(waiting_times), np.mean(turnaround_times)

# Main Execution
df = generate_tasks()
sjf_order, sjf_avg_wait, sjf_avg_turn = sjf_scheduling(df)
fcfs_order, fcfs_avg_wait, fcfs_avg_turn = fcfs_scheduling(df)
X, y = prepare_training_data(df)
model, ml_accuracy = ml_model_training(X, y)
ml_order, ml_avg_wait, ml_avg_turn = ml_scheduler(df, model)

# Output results
print("SJF Scheduling:")
print(f"Average Waiting Time: {sjf_avg_wait:.2f}")
print(f"Average Turnaround Time: {sjf_avg_turn:.2f}\n")
print("FCFS Scheduling:")
print(f"Average Waiting Time: {fcfs_avg_wait:.2f}")
print(f"Average Turnaround Time: {fcfs_avg_turn:.2f}\n")
print("ML-Based Scheduling:")
print(f"Accuracy: {ml_accuracy:.2f}")
print(f"Average Waiting Time: {ml_avg_wait:.2f}")
print(f"Average Turnaround Time: {ml_avg_turn:.2f}")