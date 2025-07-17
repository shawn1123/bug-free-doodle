# server.py
"""
FastMCP server for namespace-scoped Kubernetes introspection.
Provides deployment/pod counts, list data, and both allocated + live resource usage.
Errors are not caught—helpful for debugging access or data issues.
"""

from fastmcp import FastMCP
from kubernetes import client, config
from typing import List, Dict

# Initialize MCP without auth
mcp = FastMCP("NamespaceInspector")

# Load in-cluster Kubernetes config
config.load_incluster_config()
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()
metrics_v1 = client.CustomObjectsApi()


def parse_quantity(qty: str) -> int:
    """Convert Kubernetes CPU/memory quantities to integer units."""
    if qty.endswith("m"):
        return int(qty[:-1])
    if qty.endswith(("Ki", "Mi", "Gi")):
        unit = qty[-2:]
        num = float(qty[:-2])
        return int(num * {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3}[unit])
    return int(qty)


def extract_allocated_resources(containers) -> (Dict, Dict):
    total_req = {"cpu": 0, "memory": 0}
    total_lim = {"cpu": 0, "memory": 0}
    for c in containers:
        reqs = c.resources.requests or {}
        lims = c.resources.limits or {}
        total_req["cpu"] += parse_quantity(reqs.get("cpu", "0"))
        total_req["memory"] += parse_quantity(reqs.get("memory", "0"))
        total_lim["cpu"] += parse_quantity(lims.get("cpu", "0"))
        total_lim["memory"] += parse_quantity(lims.get("memory", "0"))
    return total_req, total_lim


@mcp.tool()
def list_deployments(namespace: str) -> List[str]:
    deps = apps_v1.list_namespaced_deployment(namespace)
    return [d.metadata.name for d in deps.items]


@mcp.tool()
def list_pods(namespace: str) -> List[str]:
    pods = core_v1.list_namespaced_pod(namespace)
    return [p.metadata.name for p in pods.items]


@mcp.tool()
def count_deployments(namespace: str) -> int:
    return len(list_deployments(namespace))


@mcp.tool()
def count_pods(namespace: str) -> int:
    return len(list_pods(namespace))


@mcp.tool()
def get_pod_resources(namespace: str, pod_name: str) -> Dict:
    pod = core_v1.read_namespaced_pod(pod_name, namespace)

    allocated_req, allocated_lim = extract_allocated_resources(pod.spec.containers)

    metrics = metrics_v1.get_namespaced_custom_object(
        "metrics.k8s.io", "v1beta1", namespace, "pods", pod_name
    )
    usage = metrics["containers"][0]["usage"]

    return {
        "pod": pod_name,
        "allocated": {
            "cpu_request_m": allocated_req["cpu"],
            "cpu_limit_m": allocated_lim["cpu"],
            "memory_request_b": allocated_req["memory"],
            "memory_limit_b": allocated_lim["memory"],
        },
        "usage": {
            "cpu": usage["cpu"],
            "memory": usage["memory"],
        },
    }


@mcp.tool()
def get_deployment_resources(namespace: str, deployment_name: str) -> Dict:
    dep = apps_v1.read_namespaced_deployment(deployment_name, namespace)
    sel = ",".join(f"{k}={v}" for k, v in dep.spec.selector.match_labels.items())
    pods = core_v1.list_namespaced_pod(namespace, label_selector=sel)

    pod_results = [get_pod_resources(namespace, p.metadata.name) for p in pods.items]

    totals = {
        "allocated": {"cpu_req_m": 0, "cpu_lim_m": 0, "mem_req_b": 0, "mem_lim_b": 0}
    }
    for pr in pod_results:
        alloc = pr["allocated"]
        totals["allocated"]["cpu_req_m"] += alloc["cpu_request_m"]
        totals["allocated"]["cpu_lim_m"] += alloc["cpu_limit_m"]
        totals["allocated"]["mem_req_b"] += alloc["memory_request_b"]
        totals["allocated"]["mem_lim_b"] += alloc["memory_limit_b"]

    return {
        "deployment": deployment_name,
        "pod_details": pod_results,
        "totals": totals,
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000, path="/mcp")
