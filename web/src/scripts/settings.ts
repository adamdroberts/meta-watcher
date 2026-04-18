type Primitive = string | number | boolean;

type ConfigDict = Record<string, Record<string, Primitive | string[]>>;

type FileDescriptor = {
  path: string;
  name: string;
  writable: boolean;
  active: boolean;
};

type FilesPayload = { active: string | null; files: FileDescriptor[] };

type SwitchPayload = {
  config: ConfigDict;
  active: string | null;
  running: boolean;
  requires_restart: boolean;
};

const RESTART_FIELDS = new Set<string>([
  "source.kind",
  "source.value",
  "source.width",
  "source.height",
  "source.fps",
  "models.provider",
  "models.mac_sam_model_id",
  "models.linux_sam_model_id",
  "models.detr_model_id",
  "models.rt_detr_model_id",
  "timings.pre_roll_seconds",
  "timings.post_roll_seconds",
  "output.directory",
  "output.recording_mode",
  "upload.enabled",
  "upload.provider",
  "upload.bucket",
  "upload.prefix",
  "upload.region",
  "upload.namespace",
  "upload.profile",
  "upload.credentials_path",
  "upload.upload_videos",
  "upload.upload_snapshots",
  "upload.upload_metadata",
  "upload.delete_after_upload",
  "upload.queue_size",
  "timestamps.enabled",
  "timestamps.ots_binary",
  "timestamps.calendar_urls",
  "timestamps.timeout_seconds",
  "timestamps.stamp_videos",
  "timestamps.stamp_snapshots",
  "timestamps.stamp_frames",
  "timestamps.stamp_metadata",
]);

const LIST_FIELDS = new Set<string>([
  "inventory.labels",
  "timestamps.calendar_urls",
]);

let lastConfig: ConfigDict = {};
let dirtyFields = new Set<string>();
let runtimeRunning = false;

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

function qs<T extends Element>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (!el) throw new Error(`Missing element ${selector}`);
  return el;
}

function allFields(): (HTMLInputElement | HTMLSelectElement)[] {
  return Array.from(
    document.querySelectorAll<HTMLInputElement | HTMLSelectElement>(
      "#settings-form [name]",
    ),
  );
}

function clearChildren(node: Element): void {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function setFieldValue(
  el: HTMLInputElement | HTMLSelectElement,
  raw: Primitive | string[] | undefined,
): void {
  const name = el.name;
  if (el instanceof HTMLInputElement && el.type === "checkbox") {
    el.checked = Boolean(raw);
    return;
  }
  if (LIST_FIELDS.has(name) && Array.isArray(raw)) {
    el.value = raw.join(", ");
    return;
  }
  el.value = raw === undefined || raw === null ? "" : String(raw);
}

function readFieldValue(
  el: HTMLInputElement | HTMLSelectElement,
): Primitive | string[] {
  const name = el.name;
  if (el instanceof HTMLInputElement && el.type === "checkbox") {
    return el.checked;
  }
  if (el instanceof HTMLInputElement && el.type === "number") {
    const n = Number(el.value);
    return Number.isFinite(n) ? n : 0;
  }
  if (LIST_FIELDS.has(name)) {
    return el.value.split(",").map((s) => s.trim()).filter(Boolean);
  }
  return el.value;
}

function populateForm(cfg: ConfigDict): void {
  lastConfig = JSON.parse(JSON.stringify(cfg));
  for (const el of allFields()) {
    const [section, key] = el.name.split(".");
    const sec = cfg[section];
    setFieldValue(el, sec ? sec[key] : undefined);
  }
  dirtyFields = new Set();
  updateRestartBanner(false);
}

function collectConfig(): ConfigDict {
  const out: ConfigDict = {};
  for (const el of allFields()) {
    const [section, key] = el.name.split(".");
    if (!out[section]) out[section] = {};
    out[section][key] = readFieldValue(el);
  }
  return out;
}

function markDirty(el: HTMLInputElement | HTMLSelectElement): void {
  const [section, key] = el.name.split(".");
  const prev = lastConfig[section]?.[key];
  const next = readFieldValue(el);
  const same = JSON.stringify(prev) === JSON.stringify(next);
  if (same) {
    dirtyFields.delete(el.name);
  } else {
    dirtyFields.add(el.name);
  }
  updateRestartBanner(
    runtimeRunning && Array.from(dirtyFields).some((f) => RESTART_FIELDS.has(f)),
  );
}

function updateRestartBanner(visible: boolean): void {
  const banner = $("restart-banner");
  banner.hidden = !visible;
}

async function loadFiles(): Promise<void> {
  const res = await fetch("/api/config/files");
  if (!res.ok) return;
  const payload: FilesPayload = await res.json();
  const select = qs<HTMLSelectElement>("#config-file");
  clearChildren(select);
  if (payload.files.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No config files discovered";
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
  }
  for (const file of payload.files) {
    const opt = document.createElement("option");
    opt.value = file.path;
    let label = file.active ? `${file.name} (active)` : file.name;
    if (!file.writable) label += " · read-only";
    opt.textContent = label;
    if (file.active) opt.selected = true;
    select.appendChild(opt);
  }
  const activeEl = $("active-path");
  activeEl.textContent = payload.active ?? "(no active file — edits only persist in-memory)";
  activeEl.title = payload.active ?? "";
}

async function loadConfig(): Promise<void> {
  const res = await fetch("/api/config");
  if (!res.ok) {
    showError(`Failed to load config: ${res.status}`);
    return;
  }
  const cfg: ConfigDict = await res.json();
  populateForm(cfg);
}

async function loadSnapshot(): Promise<void> {
  try {
    const res = await fetch("/api/snapshot");
    if (!res.ok) return;
    const body = await res.json();
    runtimeRunning = Boolean(body.running);
  } catch {
    runtimeRunning = false;
  }
  updateRestartBanner(
    runtimeRunning && Array.from(dirtyFields).some((f) => RESTART_FIELDS.has(f)),
  );
}

async function applyInMemory(): Promise<boolean> {
  const patch = collectConfig();
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: "Failed to apply" }));
    showError(body.error ?? `Failed to apply: ${res.status}`);
    return false;
  }
  const updated: ConfigDict = await res.json();
  populateForm(updated);
  return true;
}

async function saveToFile(): Promise<void> {
  if (runtimeRunning && Array.from(dirtyFields).some((f) => RESTART_FIELDS.has(f))) {
    const proceed = window.confirm(
      "Some changes require restart. Save to file now? The running pipeline will keep using the old values until Stop/Start.",
    );
    if (!proceed) return;
  }
  const applied = await applyInMemory();
  if (!applied) return;
  const res = await fetch("/api/config/save", { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: "Failed to save" }));
    showError(body.error ?? `Failed to save: ${res.status}`);
    return;
  }
  const body = await res.json();
  showToast(`Saved to ${body.path}`);
  showError(null);
  await loadFiles();
}

async function switchFile(): Promise<void> {
  const select = qs<HTMLSelectElement>("#config-file");
  const target = select.value;
  if (!target) return;
  if (runtimeRunning) {
    const proceed = window.confirm(
      "Runtime is running. Loading a different file will not restart the pipeline — the current run keeps its old config until Stop/Start. Continue?",
    );
    if (!proceed) return;
  }
  const res = await fetch("/api/config/switch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path: target }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: "Failed to switch" }));
    showError(body.error ?? `Failed to switch: ${res.status}`);
    return;
  }
  const payload: SwitchPayload = await res.json();
  populateForm(payload.config);
  showToast(`Loaded ${payload.active ?? "config"}`);
  showError(null);
  await loadFiles();
}

function showError(message: string | null): void {
  const banner = $("settings-error");
  if (!message) {
    banner.classList.remove("visible");
    banner.textContent = "";
    return;
  }
  banner.textContent = message;
  banner.classList.add("visible");
}

let toastTimer: number | null = null;
function showToast(message: string): void {
  const toast = $("settings-toast");
  toast.textContent = message;
  toast.classList.add("visible");
  if (toastTimer !== null) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("visible");
  }, 2600);
}

function init(): void {
  for (const el of allFields()) {
    el.addEventListener("input", () => markDirty(el));
    el.addEventListener("change", () => markDirty(el));
  }
  $("btn-save").addEventListener("click", () => {
    saveToFile().catch((err) => showError(String(err)));
  });
  $("btn-apply").addEventListener("click", () => {
    applyInMemory()
      .then((ok) => {
        if (ok) showToast("Applied in-memory");
      })
      .catch((err) => showError(String(err)));
  });
  $("btn-revert").addEventListener("click", () => {
    loadConfig().catch((err) => showError(String(err)));
  });
  $("btn-load-file").addEventListener("click", () => {
    switchFile().catch((err) => showError(String(err)));
  });
  Promise.all([loadFiles(), loadConfig(), loadSnapshot()]).catch((err) =>
    showError(String(err)),
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
