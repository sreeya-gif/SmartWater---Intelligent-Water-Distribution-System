import numpy as np
import networkx as nx
import heapq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random


class SmartMeter:
    def __init__(self, meter_id, location):
        self.meter_id = meter_id
        self.location = location  # (x, y) coordinates
        self.water_usage = random.randint(50, 500)  # Simulated usage in liters

    def update_usage(self):
        """ Simulates dynamic water consumption changes """
        self.water_usage = max(0, self.water_usage + random.randint(-30, 50))

meters = [SmartMeter(i, (random.randint(0, 100), random.randint(0, 100))) for i in range(10)]


X = np.array([random.randint(50, 500) for _ in range(1000)]).reshape(-1, 1)  # Previous water usage
y = np.array([max(0, x + random.randint(-30, 50)) for x in X.flatten()])  # Next predicted usage


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Model Mean Absolute Error:", mean_absolute_error(y_test, y_pred))


G = nx.Graph()


for meter in meters:
    G.add_node(meter.meter_id, pos=meter.location)


for i in range(len(meters)):
    for j in range(i + 1, len(meters)):
        if random.random() > 0.5:  # Randomly connect some meters
            distance = np.linalg.norm(np.array(meters[i].location) - np.array(meters[j].location))
            G.add_edge(meters[i].meter_id, meters[j].meter_id, weight=distance)

def dijkstra(graph, start_meter):
    """ Finds shortest paths from the start_meter to all other meters using Dijkstra's algorithm """
    pq = [(0, start_meter)]  # Priority queue: (distance, meter_id)
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_meter] = 0
    visited = set()

    while pq:
        current_dist, current_meter = heapq.heappop(pq)

        if current_meter in visited:
            continue
        visited.add(current_meter)

        for neighbor in graph.neighbors(current_meter):
            weight = graph[current_meter][neighbor]['weight']
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


optimized_routes = dijkstra(G, meters[0].meter_id)
print("Optimized Water Distribution Routes:", optimized_routes)
