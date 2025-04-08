import ray
ray.init(address='auto')  # Connect to the cluster

# Get information about the nodes in the cluster
nodes = ray.nodes()

# Print details of the cluster nodes
for node in nodes:
    print(f"Node ID: {node['NodeID']}, Alive: {node['Alive']}, Resources: {node['Resources']}")