import os
import time
import subprocess
import signal
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests

# === CONFIG ===
ISTIO_NAMESPACE = "istio-system"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_SERVICE_NAME = "prometheus-kube-prometheus-prometheus"
PROMETHEUS_PORT = 9090
REQUIRED_ISTIO_PODS = ["istiod", "istio-ingressgateway"]
METRICS_TO_QUERY = ["up", "istio_requests_total"]
METRIC_DESCRIPTIONS = {
    "up": "Target health (1=up)",
    "istio_requests_total": "Total Istio requests",
    "istio_request_duration_seconds_bucket": "Request latency (histogram)",
    "container_cpu_usage_seconds_total": "Container CPU usage",
    "container_memory_usage_bytes": "Container memory usage",
    "node_cpu_seconds_total": "Node CPU stats",
    "node_memory_MemAvailable_bytes": "Available memory on nodes",
    "prometheus_engine_query_duration_seconds_count": "Prometheus query counts"
}


def list_cluster_nodes():
    print("\n[📦] Cluster Nodes:")
    v1 = client.CoreV1Api()
    try:
        nodes = v1.list_node().items
        for node in nodes:
            name = node.metadata.name
            status = next((s.status for s in node.status.conditions if s.type == "Ready"), "Unknown")
            arch = node.status.node_info.architecture
            print(f"   - {name} | Status: {status} | Arch: {arch}")
    except Exception as e:
        print("[✗] Failed to list nodes:", e)

def list_all_pods():
    print("\n[📦] Cluster Pods:")
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_pod_for_all_namespaces().items
        for pod in pods:
            ns = pod.metadata.namespace
            name = pod.metadata.name
            status = pod.status.phase
            node = pod.spec.node_name
            print(f"   - [{ns}] {name} | Status: {status} | Node: {node}")
    except Exception as e:
        print("[✗] Failed to list pods:", e)


def load_kube_config():
    try:
        config.load_kube_config()
        print("[✓] Loaded local kubeconfig.")
    except:
        try:
            config.load_incluster_config()
            print("[✓] Loaded in-cluster kubeconfig.")
        except Exception as e:
            print("[✗] Failed to load kube config:", e)
            exit(1)

def check_istio_pods():
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace=ISTIO_NAMESPACE).items
        for required in REQUIRED_ISTIO_PODS:
            found = any(required in pod.metadata.name and pod.status.phase == "Running" for pod in pods)
            if found:
                        print(f"[✓] Istio pod '{required}' is running.")
            else:
                print(f"[✗] Istio pod '{required}' is NOT running.")
    except Exception as e:
        print(f"[✗] Error checking Istio pods: {e}")

def check_prometheus_service():
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(PROMETHEUS_SERVICE_NAME, PROMETHEUS_NAMESPACE)
        print(f"[✓] Found Prometheus service '{PROMETHEUS_SERVICE_NAME}' in namespace '{PROMETHEUS_NAMESPACE}'")
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Prometheus service '{PROMETHEUS_SERVICE_NAME}' not found in namespace '{PROMETHEUS_NAMESPACE}'")
        else:
            print(f"[✗] Error checking Prometheus service: {e}")

def start_port_forward(namespace, service_name, local_port, target_port, address="0.0.0.0"):
    print(f"[*] Starting port-forward: {address}:{local_port} -> {service_name}:{target_port}")
    cmd = [
        "kubectl", "port-forward",
        f"svc/{service_name}",
        f"{local_port}:{target_port}",
        "-n", namespace,
        "--address", address
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(2)  # allow it to start
    return proc

