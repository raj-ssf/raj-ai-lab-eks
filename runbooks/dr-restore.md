# DR restore runbook (Velero)

Recovering an app's resources from a Velero backup. Pattern works for any
namespaced workload; the worked example below uses `hello` in `default`
because that's what the lab's drill targets.

## Prereqs

1. `kubectl` context on the target cluster (e.g. `raj-ai-lab-eks`).
2. A Velero backup exists and is reachable. Confirm:
   ```bash
   kubectl -n velero get backups
   kubectl -n velero get backupstoragelocation default -o jsonpath='{.status.phase}{"\n"}'
   # Expect phase=Available
   ```
3. The app's ArgoCD Application has `selfHeal=true` — pause it BEFORE the drill
   so Velero (not ArgoCD) does the recovery and your timings reflect Velero's
   RTO, not Argo's.

## Procedure

### 1. Pause the relevant ArgoCD app's autosync

```bash
APP_NAME=hello   # whichever Application owns the resources you're drilling
kubectl -n argocd patch app $APP_NAME --type=merge \
  -p '{"spec":{"syncPolicy":{"automated":null}}}'
```

### 2. Note the start time + delete the resources

```bash
T0=$(date -u +%s)
echo "T0: $(date -u +%H:%M:%S) UTC"

# Delete EVERYTHING the app owns. Use the namespace + label both, or
# enumerate explicitly. For hello specifically:
kubectl delete deploy/hello svc/hello \
  httproute.gateway.networking.k8s.io/hello \
  -n default --wait=true

# Sanity: traffic should now fail
curl -sI --max-time 5 https://hello.ekstest.com   # expect 404 or 502
```

### 3. Submit a Velero Restore

```bash
T1=$(date -u +%s)
RESTORE_NAME="hello-drill-${T1}"   # lowercase + alphanumeric/dash only —
                                   # RFC 1123 rejects ISO 8601's `T`/`Z`

cat <<EOF | kubectl apply -f -
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: $RESTORE_NAME
  namespace: velero
spec:
  backupName: <BACKUP_NAME_FROM_velero get backups>
  includedNamespaces: ["default"]
  # Skip events — they're noise from the original backup time
  excludedResources: ["events", "events.events.k8s.io"]
  # Don't overwrite anything still in the cluster (defensive)
  existingResourcePolicy: none
EOF
```

### 4. Watch and confirm

```bash
kubectl -n velero get restore $RESTORE_NAME -w
# Wait for phase=Completed

kubectl get deploy,svc,httproute -n default
kubectl -n default rollout status deploy/hello --timeout=120s
curl -sS -o /dev/null -w "%{http_code}\n" https://hello.ekstest.com
# Expect 200
```

### 5. Re-enable ArgoCD autosync

```bash
kubectl -n argocd patch app $APP_NAME --type=merge \
  -p '{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}'
kubectl -n argocd annotate app $APP_NAME argocd.argoproj.io/refresh=hard --overwrite
```

## Drill results — 2026-04-29

Backup: `test-202604282031` (manual, ~14h old, 24h TTL, 1113 items
including all of `default`).

Resources targeted: `default/hello` (Deployment 3 replicas, Service,
HTTPRoute).

| Phase | Duration |
|---|---|
| Restore submit → Velero phase=Completed | **5–7 sec** (5 items) / 7 sec (full ns) |
| Velero done → pods Running 3/3 | ~20 sec |
| Total: T0 (delete) → HTTP 200 (verified) | **~25 sec** for clean second-attempt run |

The 25-sec floor is dominated by container start time + endpoint
propagation, NOT Velero. Velero itself is sub-10-sec for this workload
size. Expect longer RTO for apps with PVCs (Kopia fs-restore is several
seconds per GB) or init containers that pull large model artifacts.

## Gotchas hit during the drill

### Label-scope restores miss resources without the label on metadata

The first restore attempt used `spec.labelSelector: matchLabels: app=hello`
expecting it would catch every hello-owned resource. It only caught 5 of 7:
the Pods, ReplicaSet, and HTTPRoute have `app=hello` on `metadata.labels`,
but the **Deployment and Service do NOT** — the source YAML puts
`app: hello` on the Deployment's `spec.template.metadata.labels` (for the
Pods) and on the Service's `spec.selector`, but neither object has its
own `metadata.labels`. So the label-scoped restore quietly skipped them.

Symptom: HTTPRoute restored but pointed at a non-existent Service →
HTTP 500 even after restore phase=Completed.

Three ways to avoid the trap (any one suffices):

1. **Use namespace-scoped restore** (no labelSelector). Cleanest when
   the namespace contains only one app. This is what worked in the
   second attempt above.
2. **Add `app: <name>` to every `metadata.labels` block** in the source
   manifests. Tedious; brittle to forget one.
3. **Use Kustomize `commonLabels`** to inject the label on every
   resource the kustomization manages. Best practice for new apps.

### Restore name must be lowercase RFC 1123

Naming a Restore with an ISO 8601 timestamp like `hello-drill-20260429T175948Z`
fails admission — uppercase `T`/`Z` violate the lowercase-alphanumeric-and-dash
rule. Use a Unix epoch (`hello-drill-${epoch}`) or a custom lowercase format
(`hello-drill-2026-04-29-1759`).

### ArgoCD selfHeal will race Velero if not paused

With `selfHeal=true`, ArgoCD detects the resource deletion within seconds
and recreates from git, beating Velero. Your timing measures Argo's RTO,
not Velero's. Pause the app's autosync (step 1) before any DR drill.

## Reverse drill — restoring on a fresh cluster

If the cluster is being recreated from scratch (true DR scenario):

1. Stand up the new cluster + install Velero pointed at the SAME
   BackupStorageLocation (same S3 bucket).
2. `kubectl -n velero get backups` should list backups from the old
   cluster (Velero discovers them from the bucket on startup).
3. For each app, run the namespace-scoped restore from step 3 above.
4. Restore order matters when apps depend on each other (e.g.
   keycloak before anything that does OIDC). Velero respects the
   `restore` resource order via `spec.includedResources` ordering.

For the lab specifically, an end-to-end fresh-cluster recovery is a
separate drill — this runbook covers single-app/single-namespace
restore on the existing cluster.

## Drill record

### 2026-05-01 — Phase #71 backup-half drill

**Outcome**: ✅ Backup half of the procedure verified. Restore half
deferred (would briefly impact live `hello` HTTPRoute traffic;
requires operator-scheduled maintenance window).

**Procedure executed (non-destructive)**:

```bash
kubectl apply -f - <<YAML
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: phase71-drill-20260501-211619
  namespace: velero
  labels:
    drill: phase71
    target: default
spec:
  includedNamespaces: [default]
  defaultVolumesToFsBackup: false
  storageLocation: default
  ttl: 168h
YAML
```

**Measured timings**:

| Metric | Value |
|---|---|
| Backup submission → start | <1s |
| Items backed up | 274 |
| Backup duration (start → completion) | **3 seconds** |
| Errors | 0 |
| Status | Completed |
| TTL | 168h (auto-expires 2026-05-08) |

**What this proves**:

- BackupStorageLocation is reachable + the S3 path is writeable
  via the velero-pod's IAM role.
- Velero can enumerate + serialize the cluster's resources for
  `default` namespace in 3 seconds (274 items: hello deploy + svc
  + httproute + service-account + 3 pods + node-debugger remnants
  + RBAC bindings).
- The nightly-all schedule that runs at 02:00 UTC daily uses the
  exact same code path; the 7-minute duration of those backups
  reflects 2157-item cluster-wide scope, not a defect in any
  per-namespace path.

**What this DOESN'T prove (next-drill targets)**:

- That a Velero `Restore` resource can re-create the resources
  in a fresh namespace. Restore-half drill requires operator
  authorization — it would briefly delete `hello` resources +
  re-create them. RTO measurement = next drill.
- That cross-cluster restore (true DR scenario) works. That
  requires standing up a fresh cluster pointed at the same S3
  bucket, then running `velero restore create`.

**Drill artifacts**:

- Backup name: `phase71-drill-20260501-211619`
- S3 location: `s3://${cluster}-velero/backups/phase71-drill-20260501-211619/`
- Auto-expires: 2026-05-08 (TTL 168h)

**Next-drill checklist**:

1. Schedule a 30-min maintenance window where `hello.ekstest.com`
   can be 5xx for ~2-3 minutes.
2. Pause ArgoCD `hello` app autosync.
3. Run the DELETE step from "Procedure" section above against
   `default` namespace's hello resources.
4. Run `velero restore create` from the latest nightly-all backup
   targeting `--include-resources deployments,services,httproutes
   --selector app=hello`.
5. Time T0 (delete complete) → T1 (curl https://hello.ekstest.com
   returns 200 again).
6. Append the measured RTO to this runbook + record in this
   "Drill record" section.
7. Re-enable ArgoCD autosync.
