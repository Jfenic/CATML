# CATML Trusted Deployment & Threat Model Guide

> Security architecture, deployment tiers, and hardening guidelines for CATML (AutoML Platform).

---

## 1. Overview & Threat Model

CATML is architected as a **local-first, governed machine learning system**. It is designed to run securely on local workstations and private enterprise infrastructures where data privacy, experiment reproducibility, and safe AI agent operations are paramount.

When deploying CATML in networked environments, administrators and developers must adhere to the **Principle of Least Privilege** and the **Fail-Closed Default** posture.

```
                           +------------------------------------------+
                           |           Deployment Tiers               |
                           +------------------------------------------+
                                                |
               +--------------------------------+--------------------------------+
               |                                |                                |
               v                                v                                v
    [Tier 1: Local Loopback]          [Tier 2: Private LAN]           [Tier 3: Cloud VPS/Prod]
    • Host: 127.0.0.1 (default)       • Host: Private LAN IP          • Host: Private / Container
    • No token required               • Bearer token mandatory        • TLS Reverse Proxy (HTTPS)
    • Zero network exposure           • Fragment: /#token=...         • Bearer Token / mTLS
    • Safe for development            • TLS Proxy recommended         • Workspace chmod 700
```

---

## 2. Deployment Tiers

### Tier 1: Local Developer Workstation (Default & Recommended)

* **Target Environment:** Developer laptop or local workstation.
* **Network Binding:** Loopback only (`127.0.0.1` or `::1`).
* **Authentication:** Optional on loopback; no token required.
* **Security Controls:**
  * Workbench binds to `127.0.0.1:8080` by default.
  * MCP server operates via `stdio` transport (zero network ports) or `127.0.0.1:8000`.
  * Wildcard CORS headers (`*`) are strictly prohibited.
  * Zero telemetry; no data leaves the workstation.

```bash
# Local Workbench (default host is 127.0.0.1)
automl ui

# Local MCP Server via stdio (Claude Desktop / Cursor)
automl mcp --transport stdio
```

---

### Tier 2: Private LAN / Collaborative Team Server

* **Target Environment:** On-premises laboratory server or isolated team subnet.
* **Network Binding:** Private network interface (`0.0.0.0` or specific LAN IP).
* **Authentication:** **Mandatory fail-closed**.
* **Security Controls:**
  * **Ephemeral Token Generation:** If no `--auth-token` is specified when binding to a non-loopback host, Workbench automatically generates a cryptographically secure 128-bit URL-safe token.
  * **Safe Fragment Delivery:** The server displays the connection URL using the URI fragment (`/#token=<token>`). Fragments are processed client-side and **never transmitted over the wire or stored in web server access logs**.
  * **Fail-Closed Override:** Bypassing authentication on external interfaces (`--insecure-no-auth`) is strictly refused unless the environment variable `CATML_ALLOW_INSECURE=1` is explicitly set.

```bash
# Safe LAN Workbench deployment with explicit token
automl ui --host 0.0.0.0 --port 8080 --auth-token "your-team-secret-token"

# MCP Streamable-HTTP with Bearer token
automl mcp --transport streamable-http --host 0.0.0.0 --port 8000 --token "your-agent-secret"
```

---

### Tier 3: Cloud VPS & Production Hosting

* **Target Environment:** Remote VPS, Kubernetes cluster, or cloud instance.
* **Network Binding:** Internal loopback/container interface, fronted by a reverse proxy.
* **Authentication:** Multi-factor or strong Bearer authentication + TLS termination.
* **Mandatory Rules:**
  1. **NEVER expose raw HTTP or MCP ports directly to the public Internet.**
  2. Front all services with a battle-tested reverse proxy (**Caddy**, **Nginx**, or **Envoy**) terminating TLS (HTTPS / WSS).
  3. Enforce restricted firewall rules (UFW, AWS Security Groups) allowing traffic exclusively from authorized developer/agent IPs or private VPNs (WireGuard / Tailscale).
  4. Restrict workspace filesystem permissions:
     ```bash
     chmod 700 .automl/
     umask 077
     ```

#### Reference Nginx Configuration (Reverse Proxy with TLS)

```nginx
server {
    listen 443 ssl http2;
    server_name catml.internal.domain;

    ssl_certificate     /etc/ssl/certs/catml.crt;
    ssl_certificate_key /etc/ssl/private/catml.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Workbench Reverse Proxy
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # WebSocket support for live training telemetry
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # MCP Streamable-HTTP Proxy
    location /mcp {
        proxy_pass http://127.0.0.1:8000/mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Authorization $http_authorization;
    }
}
```

---

## 3. Model Artifact Security (`joblib` / `pickle`)

> [!WARNING]
> **Arbitrary Code Execution Risk**: Machine learning model serialization using Python's standard `pickle` or `joblib` libraries allows arbitrary code execution upon deserialization.

* **Never Load Untrusted Models:** Treat `.pkl` files with the same caution as executable binaries (`.exe`, `.sh`). Only load model artifacts produced by trusted, verified pipelines.
* **Integrity vs. Authenticity:** CATML generates companion `.sha256` checksum files for every exported `ModelArtifact`. These checksums guarantee **accidental data corruption detection** (integrity). They do **NOT** prove cryptographic authenticity or provenance against an adversary with write access to the filesystem.
* **Security Roadmap:**
  * **Digital Signatures:** Future releases will support optional asymmetric HMAC/PKI digital signatures (Sigstore / cosign).
  * **Safer Interchange Formats:** Inference-only pipelines will offer export to ONNX and Treelite to eliminate Python deserialization in production scoring services.

---

## 4. Agentic & MCP Attack Surface Hardening

CATML's Agent and MCP subsystems operate under the **"Propose ≠ Accept"** hypothesis-driven governance principle:

1. **Zero Raw Rows by Default:** The MCP analysis tools (such as `analysis_get_findings`) operate in compact summary mode (`compact=True`) by default. Raw dataset rows and sensitive values are excluded from LLM context windows.
2. **Deterministic Tool Allowlist:** Agents interact exclusively via typed CQRS commands and queries (`CommandBus` / `QueryBus`). They cannot execute arbitrary shell commands or spawn unrestricted sub-processes.
3. **Path Confinement:** Filesystem operations and media previews (`/api/media/preview`) are strictly confined to the active workspace and dataset roots. Path traversal attacks (`../`) are detected and rejected.
4. **Budget Leases & Cooperative Cancellation:** Agentic loops operate under token and dollar budget limits with cooperative interruption via the SQLite audit ledger.

---

## 5. Supply Chain & Dependency Hygiene

* **Vulnerability Auditing:** Maintainers and CI pipelines run `pip-audit` to detect known CVEs in upstream dependencies.
* **Dependency Pinning:** Optional deep learning extras (`torch`, `torchvision`, `optuna`) specify upper version bounds to prevent breaking API changes and unreviewed upstream modifications.
* **Integrity Verification:** Released packages are built and verified with standard SHA-256 wheel hashes.
