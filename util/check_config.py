import os
import time
import subprocess
import signal
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests


# === ENHANCED CONFIG ===
ISTIO_NAMESPACE = "istio-system"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_SERVICE_NAME = "prometheus-kube-prometheus-prometheus"
PROMETHEUS_PORT = 9090
REQUIRED_ISTIO_PODS = ["istiod", "istio-ingressgateway"]
KIALI_SERVICE_NAME = "kiali"
KIALI_NAMESPACE = "istio-system"
KIALI_LOCAL_PORT = 20001
KIALI_TARGET_PORT = 20001

# Kubernetes Dashboard configuration
K8S_DASHBOARD_NAMESPACE = "kubernetes-dashboard"
K8S_DASHBOARD_SERVICE_NAME = "kubernetes-dashboard"
K8S_DASHBOARD_LOCAL_PORT = 8443
K8S_DASHBOARD_TARGET_PORT = 8443
# Additional services to check and port-forward
SERVICES_TO_FORWARD = {
    "prometheus": {
        "namespace": "monitoring",
        "service": "prometheus-kube-prometheus-prometheus",
        "local_port": 9090,
        "target_port": 9090
    },
    "grafana": {
        "namespace": "monitoring", 
        "service": "prometheus-grafana",
        "local_port": 3000,
        "target_port": 80
    },
    "kiali": {
        "namespace": "istio-system",
        "service": "kiali",
        "local_port": 20001,
        "target_port": 20001
    },
    "jaeger": {
        "namespace": "istio-system",
        "service": "tracing",
        "local_port": 16686,
        "target_port": 80
    }
}

ONLINEBOUTIQUE_NAMESPACE = "default"  # or wherever you deploy it
REQUIRED_ONLINEBOUTIQUE_SERVICES = [
    "frontend", "cartservice", "productcatalogservice", 
    "currencyservice", "paymentservice", "shippingservice",
    "emailservice", "checkoutservice", "recommendationservice", "adservice"
]

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
    """Check if Prometheus service is available"""
    print("\n[📊] Checking Prometheus service:")
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(PROMETHEUS_SERVICE_NAME, PROMETHEUS_NAMESPACE)
        print(f"[✓] Found Prometheus service '{PROMETHEUS_SERVICE_NAME}' in namespace '{PROMETHEUS_NAMESPACE}'")
        
        # Also check if Prometheus pod is running
        pods = v1.list_namespaced_pod(namespace=PROMETHEUS_NAMESPACE).items
        prometheus_pod = next((pod for pod in pods if "prometheus-kube-prometheus-prometheus" in pod.metadata.name), None)
        if prometheus_pod and prometheus_pod.status.phase == "Running":
            print(f"[✓] Prometheus pod is running")
        else:
            print(f"[⚠️] Prometheus service found but pod may not be running")
        
        return True
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Prometheus service '{PROMETHEUS_SERVICE_NAME}' not found in namespace '{PROMETHEUS_NAMESPACE}'")
            print("    💡 To install Prometheus: helm repo add prometheus-community https://prometheus-community.github.io/helm-charts")
            print("    💡 helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace")
        else:
            print(f"[✗] Error checking Prometheus service: {e}")
        return False

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


def check_kiali_service():
    """Check if Kiali service is available"""
    print("\n[🕸️] Checking Kiali service:")
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(KIALI_SERVICE_NAME, KIALI_NAMESPACE)
        print(f"[✓] Found Kiali service '{KIALI_SERVICE_NAME}' in namespace '{KIALI_NAMESPACE}'")
        
        # Also check if Kiali pod is running
        pods = v1.list_namespaced_pod(namespace=KIALI_NAMESPACE).items
        kiali_pod = next((pod for pod in pods if "kiali" in pod.metadata.name), None)
        if kiali_pod and kiali_pod.status.phase == "Running":
            print(f"[✓] Kiali pod is running")
        else:
            print(f"[⚠️] Kiali service found but pod may not be running")
        
        return True
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Kiali service '{KIALI_SERVICE_NAME}' not found in namespace '{KIALI_NAMESPACE}'")
            print("    💡 To install Kiali: kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml")
        else:
            print(f"[✗] Error checking Kiali service: {e}")
        return False

def start_kiali_port_forward():
    """Start port forwarding for Kiali"""
    print(f"\n[🚀] Starting Kiali port-forward...")
    try:
        proc = start_port_forward(
            KIALI_NAMESPACE,
            KIALI_SERVICE_NAME,
            KIALI_LOCAL_PORT,
            KIALI_TARGET_PORT
        )
        print(f"[✓] Kiali available at: http://localhost:{KIALI_LOCAL_PORT}")
        print("    📊 Use Kiali to visualize your service mesh topology and traffic")
        return proc
    except Exception as e:
        print(f"[✗] Failed to start Kiali port-forward: {e}")
        return None

def start_prometheus_port_forward():
    """Start port forwarding for Prometheus"""
    print(f"\n[🚀] Starting Prometheus port-forward...")
    try:
        proc = start_port_forward(
            PROMETHEUS_NAMESPACE,
            PROMETHEUS_SERVICE_NAME,
            PROMETHEUS_PORT,
            PROMETHEUS_PORT
        )
        print(f"[✓] Prometheus available at: http://localhost:{PROMETHEUS_PORT}")
        print("    📊 Use Prometheus to query metrics and monitor your cluster")
        return proc
    except Exception as e:
        print(f"[✗] Failed to start Prometheus port-forward: {e}")
        return None

def check_k8s_dashboard_service():
    """Check if Kubernetes Dashboard service is available"""
    print("\n[🖥️] Checking Kubernetes Dashboard service:")
    v1 = client.CoreV1Api()
    
    # Try different possible service names and namespaces
    possible_configs = [
        ("kubernetes-dashboard", "kubernetes-dashboard"),
        ("kubernetes-dashboard", "kube-system"),
        ("dashboard", "kubernetes-dashboard"),
        ("kubernetes-dashboard-web", "kubernetes-dashboard")
    ]
    
    for service_name, namespace in possible_configs:
        try:
            svc = v1.read_namespaced_service(service_name, namespace)
            print(f"[✓] Found Kubernetes Dashboard service '{service_name}' in namespace '{namespace}'")
            
            # Update global config with found values
            global K8S_DASHBOARD_SERVICE_NAME, K8S_DASHBOARD_NAMESPACE
            K8S_DASHBOARD_SERVICE_NAME = service_name
            K8S_DASHBOARD_NAMESPACE = namespace
            
            # Check service type and provide appropriate access info
            service_type = svc.spec.type
            print(f"[ℹ️] Service type: {service_type}")
            
            if service_type == "NodePort":
                # Find the NodePort
                node_port = None
                for port in svc.spec.ports:
                    if port.node_port:
                        node_port = port.node_port
                        print(f"[ℹ️] Dashboard available via NodePort: <VM_IP>:{port.node_port}")
                        
                        # Try to get VM IP automatically
                        vm_ip = get_vm_ip()
                        if vm_ip:
                            dashboard_url = f"https://{vm_ip}:{port.node_port}"
                            print(f"[ℹ️] Auto-detected URL: {dashboard_url}")
                        else:
                            print(f"[ℹ️] Example: https://your-vm-ip:{port.node_port}")
                        print("    🔐 Note: You'll need to use your VM's external IP address")
                        break
                
                # Store node_port for later use
                if node_port:
                    global K8S_DASHBOARD_TARGET_PORT
                    K8S_DASHBOARD_TARGET_PORT = node_port
            elif service_type == "LoadBalancer":
                # Check for external IP
                if svc.status.load_balancer.ingress:
                    external_ip = svc.status.load_balancer.ingress[0].ip
                    port = svc.spec.ports[0].port
                    print(f"[ℹ️] Dashboard available via LoadBalancer: https://{external_ip}:{port}")
                else:
                    print(f"[⚠️] LoadBalancer service found but no external IP assigned yet")
            else:
                # ClusterIP - needs port forwarding
                print(f"[ℹ️] ClusterIP service - will use port-forwarding")
            
            # Also check if Dashboard pod is running
            pods = v1.list_namespaced_pod(namespace=namespace).items
            dashboard_pod = next((pod for pod in pods if "dashboard" in pod.metadata.name), None)
            if dashboard_pod and dashboard_pod.status.phase == "Running":
                print(f"[✓] Kubernetes Dashboard pod is running")
            else:
                print(f"[⚠️] Kubernetes Dashboard service found but pod may not be running")
            
            # Show token generation command
            print("    💡 Get authentication token with:")
            print(f"    💡 kubectl -n {namespace} create token kubernetes-dashboard")
            
            return True, service_type
        except ApiException:
            continue
    
    # If we get here, none of the configurations worked
    print(f"[✗] Kubernetes Dashboard not found in any expected location")
    print("    💡 To install Kubernetes Dashboard:")
    print("    💡 kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml")
    print("    💡 Alternative: Check if dashboard is installed with different name/namespace")
    print("    💡 Run: kubectl get svc --all-namespaces | grep dashboard")
    return False, None

def start_k8s_dashboard_port_forward():
    """Start port forwarding for Kubernetes Dashboard"""
    print(f"\n[🚀] Starting Kubernetes Dashboard port-forward...")
    try:
        proc = start_port_forward(
            K8S_DASHBOARD_NAMESPACE,
            K8S_DASHBOARD_SERVICE_NAME,
            K8S_DASHBOARD_LOCAL_PORT,
            K8S_DASHBOARD_TARGET_PORT
        )
        print(f"[✓] Kubernetes Dashboard available at: https://localhost:{K8S_DASHBOARD_LOCAL_PORT}")
        print("    🔐 Note: Dashboard uses HTTPS and requires authentication token")
        print("    💡 Get token with: kubectl -n kubernetes-dashboard create token kubernetes-dashboard")
        return proc
    except Exception as e:
        print(f"[✗] Failed to start Kubernetes Dashboard port-forward: {e}")
        return None
def stop_port_forward(proc, service_name):
    """Stop a specific port-forward process"""
    try:
        if proc:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            print(f"[✓] Stopped {service_name} port-forward")
    except Exception as e:
        print(f"[✗] Error stopping {service_name} port-forward: {e}")

def list_all_services():
    """List all services across all namespaces for debugging"""
    print("\n[🔍] Listing all services (for debugging):")
    v1 = client.CoreV1Api()
    try:
        services = v1.list_service_for_all_namespaces().items
        for svc in services:
            ns = svc.metadata.namespace
            name = svc.metadata.name
            type_str = svc.spec.type
            ports = [f"{p.port}:{p.target_port}" for p in svc.spec.ports] if svc.spec.ports else ["No ports"]
            print(f"   - [{ns}] {name} | Type: {type_str} | Ports: {', '.join(ports)}")
    except Exception as e:
        print(f"[✗] Failed to list services: {e}")


# ...existing code...

def get_vm_ip():
    """Try to get the VM's IP address"""
    try:
        import socket
        # Try to connect to a remote address to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except:
        return None

def open_dashboard_in_browser(url):
    """Open the dashboard URL in the default browser"""
    try:
        import webbrowser
        print(f"[🚀] Opening dashboard in browser: {url}")
        webbrowser.open(url)
        return True
    except Exception as e:
        print(f"[⚠️] Could not open browser automatically: {e}")
        return False

def main():
    """Main function to test Kiali, Prometheus and Dashboard functionality"""
    print("🔧 Kubernetes Cluster Checker - Kiali, Prometheus & Dashboard Test")
    print("=" * 70)
    
    # Load kubeconfig
    load_kube_config()
    
    # Check basic cluster info
    list_cluster_nodes()
    
    # Check Istio components
    print("\n[🔍] Checking Istio components:")
    check_istio_pods()
    
    # Check services
    kiali_available = check_kiali_service()
    prometheus_available = check_prometheus_service()
    dashboard_available, dashboard_service_type = check_k8s_dashboard_service()
    
    # If dashboard not found, offer to list all services for debugging
    if not dashboard_available:
        response = input("\nDashboard not found. List all services for debugging? (y/N): ")
        if response.lower() == 'y':
            list_all_services()
    
    # Manage port-forwards
    active_processes = {}
    
    if kiali_available:
        response = input(f"\nStart Kiali port-forward on localhost:{KIALI_LOCAL_PORT}? (y/N): ")
        if response.lower() == 'y':
            kiali_proc = start_kiali_port_forward()
            if kiali_proc:
                active_processes['kiali'] = kiali_proc
    
    if prometheus_available:
        response = input(f"\nStart Prometheus port-forward on localhost:{PROMETHEUS_PORT}? (y/N): ")
        if response.lower() == 'y':
            prometheus_proc = start_prometheus_port_forward()
            if prometheus_proc:
                active_processes['prometheus'] = prometheus_proc
    
    # Only offer port-forwarding for dashboard if it's ClusterIP
    if dashboard_available and dashboard_service_type == "ClusterIP":
        response = input(f"\nStart Kubernetes Dashboard port-forward on localhost:{K8S_DASHBOARD_LOCAL_PORT}? (y/N): ")
        if response.lower() == 'y':
            dashboard_proc = start_k8s_dashboard_port_forward()
            if dashboard_proc:
                active_processes['dashboard'] = dashboard_proc
    elif dashboard_available and dashboard_service_type == "NodePort":
        print(f"\n[ℹ️] Dashboard is accessible directly via NodePort - no port-forwarding needed")
        # Offer to open dashboard in browser
        vm_ip = get_vm_ip()
        if vm_ip:
            dashboard_url = f"https://{vm_ip}:{K8S_DASHBOARD_TARGET_PORT}"
            response = input(f"\nOpen Kubernetes Dashboard in browser ({dashboard_url})? (y/N): ")
            if response.lower() == 'y':
                if open_dashboard_in_browser(dashboard_url):
                    print(f"[✓] Dashboard opened in browser")
                    print(f"[ℹ️] You'll need to get a token with:")
                    print(f"[ℹ️] kubectl -n {K8S_DASHBOARD_NAMESPACE} create token kubernetes-dashboard")
                else:
                    print(f"[ℹ️] Please manually open: {dashboard_url}")
        else:
            print(f"[ℹ️] Could not auto-detect VM IP. Please access dashboard manually.")
    elif dashboard_available:
        print(f"\n[ℹ️] Dashboard is accessible directly via {dashboard_service_type} - no port-forwarding needed")
    
    if active_processes:
        print(f"\n[ℹ️] Active services:")
        if 'kiali' in active_processes:
            print(f"   • Kiali: http://localhost:{KIALI_LOCAL_PORT}")
        if 'prometheus' in active_processes:
            print(f"   • Prometheus: http://localhost:{PROMETHEUS_PORT}")
        if 'dashboard' in active_processes:
            print(f"   • Kubernetes Dashboard: https://localhost:{K8S_DASHBOARD_LOCAL_PORT}")
        
        try:
            print("\n[ℹ️] Press Ctrl+C to stop all port-forwards...")
            signal.pause()
        except KeyboardInterrupt:
            print("\n[🛑] Stopping all port-forwards...")
            for service_name, proc in active_processes.items():
                stop_port_forward(proc, service_name.title())
            print("[👋] Cleanup complete!")
    else:
        print("\n[ℹ️] No services are available or started.")

if __name__ == "__main__":
    main()