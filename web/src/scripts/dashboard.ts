interface EventSummary {
  event_id: string;
  clip_key: string | null;
  overlay_clip_key: string | null;
  snapshot_key: string | null;
  metadata_key: string | null;
  frame_count: number;
  total_size: number;
  earliest_modified: string | null;
  latest_modified: string | null;
  timestamped_keys: string[];
}

interface ListResponse {
  enabled: boolean;
  provider: string;
  bucket?: string;
  prefix?: string;
  events: EventSummary[];
  error?: string;
}

interface ArtifactInfo {
  key: string;
  kind: string;
  size: number;
  time_modified: string | null;
  sidecar_key: string | null;
}

interface DetailResponse {
  event_id: string;
  metadata: Record<string, unknown>;
  metadata_key: string | null;
  artifacts: ArtifactInfo[];
}

interface VerifyResult {
  key: string;
  status: string;
  message: string;
}

interface VerifyResponse {
  event_id: string;
  results: VerifyResult[];
}

const GRID         = document.getElementById("recording-grid")          as HTMLDivElement;
const DETAIL       = document.getElementById("recording-detail")        as HTMLElement;
const DETAIL_BODY  = document.getElementById("recording-detail-body")   as HTMLDivElement;
const DETAIL_CLOSE = document.getElementById("recording-detail-close")  as HTMLButtonElement;
const BANNER       = document.getElementById("dashboard-banner")        as HTMLDivElement;
const BADGE        = document.getElementById("dashboard-storage-badge") as HTMLSpanElement;
const REFRESH      = document.getElementById("dashboard-refresh")       as HTMLButtonElement;
const MAIN         = document.querySelector(".dashboard-main")          as HTMLElement;

function emptyNote(text: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "recording-empty";
  el.textContent = text;
  return el;
}

function setBanner(text: string, visible: boolean, variant: "" | "warn" = ""): void {
  BANNER.textContent = text;
  BANNER.className = "banner" + (visible ? " visible" : "") + (variant ? " " + variant : "");
}

function artifactUrl(eventId: string, key: string): string {
  return `/api/recordings/${encodeURIComponent(eventId)}/artifact?key=${encodeURIComponent(key)}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024)                 return `${bytes} B`;
  if (bytes < 1024 * 1024)          return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)   return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "short", timeStyle: "medium" });
}

async function loadHealth(): Promise<void> {
  try {
    const res = await fetch("/api/storage/health");
    const data = await res.json();
    if (data.ok) {
      BADGE.textContent = `storage: ${data.provider} ok`;
      BADGE.dataset.state = "ok";
    } else {
      BADGE.textContent = `storage: ${data.reason ?? "unavailable"}`;
      BADGE.dataset.state = "error";
    }
  } catch {
    BADGE.textContent = "storage: unreachable";
    BADGE.dataset.state = "error";
  }
}

async function loadList(): Promise<void> {
  GRID.replaceChildren(emptyNote("Loading…"));
  try {
    const res = await fetch("/api/recordings");
    if (!res.ok) {
      setBanner(`Failed to load recordings (${res.status}).`, true);
      GRID.replaceChildren();
      return;
    }
    const data: ListResponse = await res.json();
    if (!data.enabled) {
      setBanner("Upload is disabled in config — no recordings to browse.", true, "warn");
      GRID.replaceChildren();
      return;
    }
    setBanner(data.error ?? "", Boolean(data.error));
    renderGrid(data.events);
  } catch (err) {
    setBanner(`Error loading recordings: ${(err as Error).message}`, true);
  }
}

function renderGrid(events: EventSummary[]): void {
  if (events.length === 0) {
    GRID.replaceChildren(emptyNote("No recordings found in the configured bucket."));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const evt of events) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "recording-card";
    card.dataset.eventId = evt.event_id;
    card.setAttribute("aria-selected", "false");

    const thumb = document.createElement("span");
    thumb.className = "thumb";
    if (evt.snapshot_key) {
      thumb.style.backgroundImage = `url(${artifactUrl(evt.event_id, evt.snapshot_key)})`;
    }
    card.appendChild(thumb);

    const body = document.createElement("div");
    body.className = "body";

    const eid = document.createElement("span");
    eid.className = "event-id";
    eid.textContent = evt.event_id;
    body.appendChild(eid);

    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";
    metaRow.textContent =
      `${formatTime(evt.latest_modified)} · ${formatSize(evt.total_size)} · ` +
      `${evt.frame_count} frame${evt.frame_count === 1 ? "" : "s"}`;
    body.appendChild(metaRow);

    const badgeRow = document.createElement("div");
    badgeRow.className = "meta-row";
    const hasTs = evt.timestamped_keys.length > 0;
    const badge = document.createElement("span");
    badge.className = "verify-badge";
    badge.dataset.status = hasTs ? "pending" : "none";
    badge.textContent = hasTs
      ? `timestamped ×${evt.timestamped_keys.length}`
      : "no timestamps";
    badgeRow.appendChild(badge);
    body.appendChild(badgeRow);

    card.appendChild(body);
    card.addEventListener("click", () => selectCard(card, evt.event_id));
    frag.appendChild(card);
  }
  GRID.replaceChildren(frag);
}

let activeCard: HTMLElement | null = null;

async function selectCard(card: HTMLElement, eventId: string): Promise<void> {
  if (activeCard) activeCard.setAttribute("aria-selected", "false");
  card.setAttribute("aria-selected", "true");
  activeCard = card;
  DETAIL.hidden = false;
  MAIN.classList.add("detail-open");
  DETAIL_BODY.replaceChildren(emptyNote("Loading…"));
  try {
    const res = await fetch(`/api/recordings/${encodeURIComponent(eventId)}`);
    if (!res.ok) {
      DETAIL_BODY.replaceChildren(emptyNote(`Failed to load details (${res.status}).`));
      return;
    }
    renderDetail(await res.json());
  } catch (err) {
    DETAIL_BODY.replaceChildren(emptyNote(`Error: ${(err as Error).message}`));
  }
}

function renderDetail(detail: DetailResponse): void {
  const eid = detail.event_id;
  const clip     = detail.artifacts.find((a) => a.kind === "clip");
  const overlay  = detail.artifacts.find((a) => a.kind === "overlay_clip");
  const snapshot = detail.artifacts.find((a) => a.kind === "snapshot");
  const frames   = detail.artifacts.filter((a) => a.kind === "frame");

  const h2 = document.createElement("h2");
  h2.textContent = eid;

  const verifyBtn = document.createElement("button");
  verifyBtn.className = "primary";
  verifyBtn.type = "button";
  verifyBtn.textContent = "Verify timestamps";
  verifyBtn.addEventListener("click", () => runVerify(eid, verifyBtn));

  const pieces: Node[] = [h2, verifyBtn];

  if (clip) {
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = artifactUrl(eid, clip.key);
    pieces.push(video);
  }

  if (overlay && overlay.key !== clip?.key) {
    const label = document.createElement("div");
    label.className = "meta-row";
    label.textContent = "Overlay clip:";
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = artifactUrl(eid, overlay.key);
    pieces.push(label, video);
  }

  if (snapshot) {
    const label = document.createElement("div");
    label.className = "meta-row";
    label.textContent = "Event snapshot:";
    const img = document.createElement("img");
    img.src = artifactUrl(eid, snapshot.key);
    img.alt = `Snapshot for ${eid}`;
    pieces.push(label, img);
  }

  if (Object.keys(detail.metadata).length > 0) {
    const label = document.createElement("div");
    label.className = "meta-row";
    label.textContent = "Metadata:";
    pieces.push(label, renderMetadataTable(detail.metadata));
  }

  if (frames.length > 0) {
    const label = document.createElement("div");
    label.className = "meta-row";
    label.textContent = `Live frames (${frames.length}):`;
    pieces.push(label);
    const gallery = document.createElement("div");
    gallery.className = "gallery";
    for (const f of frames) {
      const img = document.createElement("img");
      img.src = artifactUrl(eid, f.key);
      img.alt = f.key;
      img.loading = "lazy";
      img.addEventListener("click", () =>
        window.open(artifactUrl(eid, f.key), "_blank"),
      );
      gallery.appendChild(img);
    }
    pieces.push(gallery);
  }

  const artifactsLabel = document.createElement("div");
  artifactsLabel.className = "meta-row";
  artifactsLabel.textContent = "All artifacts:";
  pieces.push(artifactsLabel);

  const list = document.createElement("div");
  list.className = "artifact-list";
  for (const a of detail.artifacts) {
    const row = document.createElement("div");
    row.className = "row";

    const kind = document.createElement("span"); kind.className = "kind"; kind.textContent = a.kind;
    const key = document.createElement("span");  key.className  = "key";  key.textContent = a.key;
    const size = document.createElement("span"); size.textContent = formatSize(a.size);

    const badge = document.createElement("span");
    badge.className = "verify-badge";
    badge.dataset.status = a.sidecar_key ? "pending" : "none";
    badge.dataset.artifactKey = a.key;
    badge.textContent = a.sidecar_key ? "timestamped" : "no ots";

    const dl = document.createElement("a");
    dl.href = artifactUrl(eid, a.key);
    dl.textContent = "download";
    const filename = a.key.split("/").pop();
    if (filename) dl.setAttribute("download", filename);

    row.append(kind, key, size, badge, dl);
    list.appendChild(row);
  }
  pieces.push(list);

  DETAIL_BODY.replaceChildren(...pieces);
}

function renderMetadataTable(meta: Record<string, unknown>): HTMLTableElement {
  const tbl = document.createElement("table");
  tbl.className = "metadata-table";
  for (const [k, v] of Object.entries(meta)) {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = k;
    const td = document.createElement("td");
    td.textContent =
      typeof v === "object" && v !== null ? JSON.stringify(v, null, 2) : String(v);
    tr.append(th, td);
    tbl.appendChild(tr);
  }
  return tbl;
}

async function runVerify(eventId: string, btn: HTMLButtonElement): Promise<void> {
  btn.disabled = true;
  btn.textContent = "Verifying…";
  try {
    const res = await fetch(
      `/api/recordings/${encodeURIComponent(eventId)}/verify`,
      { method: "POST" },
    );
    const data: VerifyResponse = await res.json();
    if (!res.ok) {
      setBanner(
        `Verify failed: ${(data as { error?: string }).error ?? res.statusText}`,
        true,
      );
      return;
    }
    applyVerifyResults(data.results);
  } catch (err) {
    setBanner(`Verify error: ${(err as Error).message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Verify timestamps";
  }
}

function applyVerifyResults(results: VerifyResult[]): void {
  const byKey = new Map(results.map((r) => [r.key, r]));
  const badges = DETAIL_BODY.querySelectorAll<HTMLSpanElement>(
    ".verify-badge[data-artifact-key]",
  );
  for (const b of badges) {
    const key = b.dataset.artifactKey;
    if (!key) continue;
    const r = byKey.get(key);
    if (!r) continue;
    b.dataset.status = r.status;
    b.textContent = r.status.replace("_", " ");
    b.title = r.message;
  }
}

function closeDetail(): void {
  DETAIL.hidden = true;
  MAIN.classList.remove("detail-open");
  if (activeCard) activeCard.setAttribute("aria-selected", "false");
  activeCard = null;
}

DETAIL_CLOSE.addEventListener("click", closeDetail);
REFRESH.addEventListener("click", () => {
  loadHealth();
  loadList();
});

document.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  loadList();
});
