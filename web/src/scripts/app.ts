type ConfigShape = {
  source: { kind: string; value: string };
  thresholds: { person_confidence: number; inventory_confidence: number };
  inventory: { auto_rescan: boolean; labels: string[] };
  output: { directory: string };
};

type WebcamDevice = { value: string; label: string; path: string | null };

type SnapshotPayload = {
  running: boolean;
  mode: string;
  status_text: string;
  recording_active: boolean;
  inventory_items: Array<{ label: string; confidence: number; samples: number }>;
  completed_clips: string[];
  error: string | null;
};

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

function input(name: string): HTMLInputElement | HTMLSelectElement {
  const el = document.querySelector<HTMLInputElement | HTMLSelectElement>(
    `[name="${name}"]`,
  );
  if (!el) throw new Error(`Missing input [name=${name}]`);
  return el;
}

async function loadConfig(): Promise<void> {
  const res = await fetch("/api/config");
  if (!res.ok) return;
  const cfg: ConfigShape = await res.json();
  (input("source_kind") as HTMLSelectElement).value = cfg.source.kind;
  (input("source_value") as HTMLInputElement).value = cfg.source.value;
  (input("output_dir") as HTMLInputElement).value = cfg.output.directory;
  (input("person_threshold") as HTMLInputElement).value = String(cfg.thresholds.person_confidence);
  (input("inventory_threshold") as HTMLInputElement).value = String(cfg.thresholds.inventory_confidence);
  (input("inventory_labels") as HTMLInputElement).value = (cfg.inventory.labels ?? []).join(", ");
  (input("auto_rescan") as HTMLInputElement).checked = cfg.inventory.auto_rescan;
  syncSourceInputs();
}

async function loadWebcams(preserve: string | null = null): Promise<void> {
  const select = input("webcam_device") as HTMLSelectElement;
  const currentValue =
    preserve ?? (input("source_value") as HTMLInputElement).value.trim();
  clearChildren(select);
  try {
    const res = await fetch("/api/devices/webcams");
    if (!res.ok) return;
    const payload: { devices: WebcamDevice[] } = await res.json();
    for (const device of payload.devices) {
      const opt = document.createElement("option");
      opt.value = device.value;
      opt.textContent = device.label;
      select.appendChild(opt);
    }
    if (payload.devices.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No cameras detected";
      opt.disabled = true;
      opt.selected = true;
      select.appendChild(opt);
    } else if (currentValue && payload.devices.some((d) => d.value === currentValue)) {
      select.value = currentValue;
    } else {
      select.value = payload.devices[0].value;
      (input("source_value") as HTMLInputElement).value = payload.devices[0].value;
    }
  } catch {
    // leave select empty on failure; user can still type an index manually.
  }
}

function syncSourceInputs(): void {
  const kind = (input("source_kind") as HTMLSelectElement).value;
  const select = input("webcam_device") as HTMLSelectElement;
  const text = input("source_value") as HTMLInputElement;
  const refresh = document.getElementById("btn-refresh-cams");
  if (kind === "webcam") {
    select.style.display = "";
    text.style.display = "none";
    if (refresh) refresh.style.display = "";
    loadWebcams(text.value.trim() || null).catch(() => {});
  } else {
    select.style.display = "none";
    text.style.display = "";
    if (refresh) refresh.style.display = "none";
  }
}

function resolvedSourceValue(): string {
  const kind = (input("source_kind") as HTMLSelectElement).value;
  if (kind === "webcam") {
    const select = input("webcam_device") as HTMLSelectElement;
    if (select.value) return select.value;
  }
  return (input("source_value") as HTMLInputElement).value.trim();
}

function collectPatch(): Record<string, unknown> {
  return {
    source: {
      kind: (input("source_kind") as HTMLSelectElement).value,
      value: resolvedSourceValue(),
    },
    thresholds: {
      person_confidence: Number((input("person_threshold") as HTMLInputElement).value),
      inventory_confidence: Number((input("inventory_threshold") as HTMLInputElement).value),
    },
    inventory: {
      auto_rescan: (input("auto_rescan") as HTMLInputElement).checked,
      labels: (input("inventory_labels") as HTMLInputElement).value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    },
    output: {
      directory: (input("output_dir") as HTMLInputElement).value.trim() || "recordings",
    },
  };
}

async function startRuntime(event: SubmitEvent): Promise<void> {
  event.preventDefault();
  const patch = collectPatch();
  await fetch("/api/config", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch),
  });
  await fetch("/api/runtime/recording", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled: (input("recording_enabled") as HTMLInputElement).checked }),
  });
  const res = await fetch("/api/runtime/start", { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: "Failed to start" }));
    showError(body.error ?? "Failed to start");
    return;
  }
  showError(null);
  reloadStream();
}

async function stopRuntime(): Promise<void> {
  await fetch("/api/runtime/stop", { method: "POST" });
}

async function manualRescan(): Promise<void> {
  await fetch("/api/runtime/rescan", { method: "POST" });
}

function reloadStream(): void {
  const img = $("video-stream") as HTMLImageElement;
  // Bust the browser cache so a stopped -> started cycle reconnects.
  img.src = `/stream.mjpg?t=${Date.now()}`;
}

function showError(message: string | null): void {
  const banner = $("error-banner");
  if (!message) {
    banner.classList.remove("visible");
    banner.textContent = "";
    return;
  }
  banner.textContent = message;
  banner.classList.add("visible");
}

function clearChildren(node: Element): void {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function renderSnapshot(snapshot: SnapshotPayload): void {
  const modeEl = $("status-mode");
  modeEl.textContent = snapshot.mode;
  modeEl.dataset.mode = snapshot.mode;
  $("status-text").textContent = snapshot.status_text || "idle";
  $("status-recording").textContent = snapshot.recording_active ? "active" : "idle";

  const list = $("inventory-list");
  clearChildren(list);
  for (const item of snapshot.inventory_items) {
    const li = document.createElement("li");
    li.textContent = `${item.label} (${item.confidence.toFixed(2)}, n=${item.samples})`;
    list.appendChild(li);
  }

  const clips = $("completed-clips");
  clips.textContent = snapshot.completed_clips.join("\n");

  ($("btn-start") as HTMLButtonElement).disabled = snapshot.running;
  ($("btn-stop") as HTMLButtonElement).disabled = !snapshot.running;
  ($("btn-rescan") as HTMLButtonElement).disabled = !snapshot.running;

  showError(snapshot.error);
}

async function pollLoop(): Promise<void> {
  while (true) {
    try {
      const res = await fetch("/api/snapshot");
      if (res.ok) {
        const payload: SnapshotPayload = await res.json();
        renderSnapshot(payload);
      }
    } catch {
      // network hiccups during server restart — retry next tick
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

function init(): void {
  $("controls").addEventListener("submit", (e) => startRuntime(e as SubmitEvent));
  $("btn-stop").addEventListener("click", stopRuntime);
  $("btn-rescan").addEventListener("click", manualRescan);
  (input("source_kind") as HTMLSelectElement).addEventListener("change", syncSourceInputs);
  (input("webcam_device") as HTMLSelectElement).addEventListener("change", (e) => {
    const target = e.target as HTMLSelectElement;
    (input("source_value") as HTMLInputElement).value = target.value;
  });
  const refresh = document.getElementById("btn-refresh-cams");
  if (refresh) refresh.addEventListener("click", () => loadWebcams(null));
  loadConfig().catch(() => {});
  pollLoop().catch(() => {});
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
