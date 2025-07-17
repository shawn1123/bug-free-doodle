# server.py
"""
FastMCP-based MCP server for namespace-scoped Kubernetes introspection.
Provides deployment and pod metrics within a specified namespace.
Includes OAuth2-based authentication and robust error handling.
"""

import os
from fastmcp import FastMCP
from fastmcp.client.auth.oauth import OAuth  # OAuth support
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from typing import List, Dict

# Load OAuth configuration from environment variables (configured in OpenShift)
OAUTH_URL = os.getenv("OAUTH_URL")
OAUTH_SCOPES = os.getenv("OAUTH_SCOPES", "").split(",")  # e.g. "openid,profile"
if not OAUTH_URL:
    raise RuntimeError("Missing OAUTH_URL environment variable")

# Create MCP server with OAuth auth guard for HTTP transport
mcp = FastMCP("NamespaceInspector", auth=OAuth(OAUTH_URL, scopes=OAUTH_SCOPES))

# Initialize Kubernetes client inside cluster (OpenShift)
config.load_incluster_config()
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()
metrics_v1 = client.CustomObjectsApi()


def safe_api_call(func, *args, default=None, **kwargs):
    """Wrapper to call Kubernetes API safely with error handling."""
    try:
        return func(*args, **kwargs)
    except ApiException as e:
        # 404 or 403 indicate unauthorized/not found; return default
        print(f"API error ({e.status}): {e.reason} for {func.__name__}")
        return default
    except Exception as e:
        # Unexpected error
        print(f"Unexpected error in {func.__name__}: {e}")
        return default


@mcp.tool()
def list_deployments(namespace: str) -> List[str]:
    """Return list of deployment names in the namespace."""
    deps = safe_api_call(apps_v1.list_namespaced_deployment, namespace, default=client.V1DeploymentList(items=[]))
    return [d.metadata.name for d in deps.items]


@mcp.tool()
def list_pods(namespace: str) -> List[str]:
    """Return list of pod names in the namespace."""
    pods = safe_api_call(core_v1.list_namespaced_pod, namespace, default=client.V1PodList(items=[]))
    return [p.metadata.name for p in pods.items]


@mcp.tool()
def count_deployments(namespace: str) -> int:
    """Return number of deployments in the namespace."""
    return len(list_deployments(namespace))


@mcp.tool()
def count_pods(namespace: str) -> int:
    """Return number of pods in the namespace."""
    return len(list_pods(namespace))


@mcp.tool()
def get_deployment_resources(namespace: str, deployment_name: str) -> Dict:
    """
    Return CPU and memory usage for all pods belonging to a given deployment.
    Metrics are gathered from the Kubernetes metrics API.
    """
    dep = safe_api_call(apps_v1.read_namespaced_deployment, deployment_name, namespace, default=None)
    if not dep:
        return {"error": f"Deployment {deployment_name} not found in {namespace}"}

    # Label-selector based on deployment spec
    labels = dep.spec.selector.match_labels
    sel = ",".join([f"{k}={v}" for k, v in labels.items()])
    pods = safe_api_call(core_v1.list_namespaced_pod, namespace, label_selector=sel,
                         default=client.V1PodList(items=[]))
    usages = []
    for p in pods.items:
        data = safe_api_call(
            metrics_v1.get_namespaced_custom_object,
            "metrics.k8s.io", "v1beta1", namespace, "pods", p.metadata.name, default=None
        )
        if data and "containers" in data:
            cpu = data["containers"][0]["usage"]["cpu"]
            mem = data["containers"][0]["usage"]["memory"]
        else:
            cpu = mem = None
        usages.append({"pod": p.metadata.name, "cpu": cpu, "memory": mem})

    return {"deployment": deployment_name, "pods": usages}


@mcp.tool()
def get_pod_resources(namespace: str, pod_name: str) -> Dict:
    """
    Return CPU and memory usage for a specific pod.
    """
    data = safe_api_call(
        metrics_v1.get_namespaced_custom_object,
        "metrics.k8s.io", "v1beta1", namespace, "pods", pod_name, default=None
    )
    if not data or "containers" not in data:
        return {"error": f"Metrics not available for pod {pod_name} in {namespace}"}
    cpu = data["containers"][0]["usage"]["cpu"]
    mem = data["containers"][0]["usage"]["memory"]
    return {"pod": pod_name, "cpu": cpu, "memory": mem}


if __name__ == "__main__":
    # Start MCP server with HTTP transport on specified port
    mcp.run(transport="http", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), path="/mcp")
