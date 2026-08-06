import csv
import os
import re
import tempfile

# repo id + revision are user-entered. Empty = not configured yet.
# Kept defined (empty) so the imports elsewhere don't break.
HF_REPO_ID = ""
HF_DATASET_REVISION = ""
# A curated manifest shipped in a dataset repo is matched by this pattern, so a
# repo can name it manifest.csv or manifest_v44.csv, etc.
MANIFEST_REPO_PATTERN = re.compile(r"(?i)^manifest.*\.csv$")
# Naming convention linking a dataset .glb to the mesh name the model emits.
GLB_SUFFIX = "_draco.glb"
MESH_SUFFIX = "_align.vtk"
MODULE_DIR = os.path.dirname(__file__)

# Available-column states
STATE_YES = "Yes"
STATE_NO_LOCAL = "No - select matching library"
STATE_NO_REPO = "Enter a dataset repo"
STATE_CACHED = "Cached"
STATE_DOWNLOAD = "Will download"
STATE_MISSING = "Missing from dataset"
STATE_AUTH = "Auth required"
STATE_NETWORK = "Network error"

# Error taxonomy
CATEGORY_AUTH = "auth"
CATEGORY_NETWORK = "network"
CATEGORY_MISSING = "missing"
CATEGORY_DECODE = "decode"


def compute_lookup_key(mesh_name):
    # Same derivation used to build the manifest. Idempotent: key(key(x)) == key(x).
    key = os.path.basename(mesh_name).lower()
    for ext in (".vtk", ".glb", ".ply"):
        if key.endswith(ext):
            key = key[:-len(ext)]
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    while True:
        stripped = False
        for suffix in ("_align", "_draco"):
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                stripped = True
        if not stripped:
            break
    return key


class Resolution:
    def __init__(self, kind, value=None, reason="", state=None):
        self.kind = kind      # 'path' | 'missing'
        self.value = value    # local file path or None
        self.reason = reason  # human message; never a token
        self.state = state    # Available-column state


class ReferenceManifest:
    def __init__(self, rows):
        self._by_key = {}
        for row in rows:
            self._by_key[row["lookup_key"]] = row

    @classmethod
    def load(cls, path):
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return cls(rows)

    @classmethod
    def from_hf_paths(cls, hf_paths):
        # Build the name -> file map from the listing itself, for repos that
        # follow the naming convention but ship no manifest.csv. Metadata columns
        # stay blank here - only lookup_key + hf_path actually drive resolution.
        rows = []
        seen = set()
        for hf_path in hf_paths:
            if not hf_path.endswith(GLB_SUFFIX):
                continue
            mesh_name = hf_path[:-len(GLB_SUFFIX)] + MESH_SUFFIX
            key = compute_lookup_key(mesh_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "lookup_key": key, "mesh_name": mesh_name, "hf_path": hf_path,
                "match": "", "family": "", "genus": "", "species": "",
                "specimen": "", "vertebra": "",
            })
        return cls(rows)

    def rows(self):
        return list(self._by_key.values())

    def lookup(self, mesh_name):
        return self._by_key.get(compute_lookup_key(mesh_name))


def _pick_manifest_file(files):
    # Choose a root-level curated manifest from a repo listing: an exact
    # manifest.csv wins, else the first manifest*.csv (e.g. manifest_v44.csv).
    candidates = [f for f in files if "/" not in f and MANIFEST_REPO_PATTERN.match(f)]
    if not candidates:
        return None
    for f in candidates:
        if f.lower() == "manifest.csv":
            return f
    return sorted(candidates)[0]


# Token resolution (Qt-free)

def read_token_file(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as handle:
            token = handle.read().strip()
    except OSError:
        return None
    return token or None


def _default_cached_token():
    # huggingface_hub already resolves env + the OS CLI-login cache; use it directly.
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def resolve_hf_token(env=None, stored_path=None, cached_token_provider=None):
    env = env if env is not None else os.environ
    token = (env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token:
        return token
    token = read_token_file(stored_path)
    if token:
        return token
    provider = cached_token_provider or _default_cached_token
    token = provider()
    if token:
        return token.strip()
    raise ValueError(
        "No HuggingFace token found. Run 'huggingface-cli login', set HF_TOKEN, "
        "or select a token file. The dataset is gated, so a token is required."
    )


# Error taxonomy

def map_download_error(exc):
    name = type(exc).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        return CATEGORY_AUTH
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if name == "HfHubHTTPError" and status is not None:
        status = int(status)
        if status in (401, 403):
            return CATEGORY_AUTH
        if status == 429:
            return CATEGORY_NETWORK
        if 500 <= status < 600:
            return CATEGORY_NETWORK
    if name in ("Timeout", "ConnectTimeout", "ReadTimeout", "ConnectionError"):
        return CATEGORY_NETWORK
    return CATEGORY_MISSING


def decode_glb_to_vtk(glb_path, cache_key):
    # Draco .glb -> temp .vtk; Slicer's loadModel cannot read Draco directly.
    import draco_glb
    poly_data = draco_glb.glb_to_polydata(glb_path)
    out_dir = os.path.join(tempfile.gettempdir(), "fossilnsm_reference")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cache_key + ".vtk")
    draco_glb.write_vtk(poly_data, out_path)
    return out_path


# Backends

class ReferenceMeshBackend:
    def resolve(self, mesh_name):
        raise NotImplementedError

    def availability(self, mesh_name):
        # Cheap, no-network state for the Available column.
        raise NotImplementedError


class LocalFolderBackend(ReferenceMeshBackend):
    MESH_EXTS = (".vtk", ".vtp", ".ply", ".stl", ".obj", ".glb")

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._index = None
        self._decode_cache = {}

    def _build_index(self):
        # Map lookup_key -> file path, so a match resolves regardless of the file's
        # extension or casing (e.g. aligned _align.ply vs the manifest's _align.vtk).
        import glob
        index = {}
        if self.root_dir and os.path.isdir(self.root_dir):
            for path in glob.glob(os.path.join(self.root_dir, "**", "*"), recursive=True):
                if os.path.splitext(path)[1].lower() in self.MESH_EXTS:
                    index.setdefault(compute_lookup_key(os.path.basename(path)), path)
        return index

    def _find(self, mesh_name):
        if not self.root_dir:
            return None
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(compute_lookup_key(mesh_name))

    def resolve(self, mesh_name):
        path = self._find(mesh_name)
        if not path:
            return Resolution("missing", None,
                              "Reference mesh is not available: {}".format(mesh_name), STATE_NO_LOCAL)
        if path.lower().endswith(".glb"):
            cached = self._decode_cache.get(mesh_name)
            if cached and os.path.isfile(cached):
                return Resolution("path", cached, state=STATE_YES)
            try:
                path = decode_glb_to_vtk(path, compute_lookup_key(mesh_name))
            except Exception as e:
                return Resolution("missing", None,
                                  "Could not decode {}: {}".format(mesh_name, e), STATE_NO_LOCAL)
            self._decode_cache[mesh_name] = path
        return Resolution("path", path, state=STATE_YES)

    def availability(self, mesh_name):
        return STATE_YES if self._find(mesh_name) else STATE_NO_LOCAL


class HuggingFaceBackend(ReferenceMeshBackend):
    def __init__(self, repo_id=None, revision=None, token_provider=None, manifest=None):
        self.repo_id = (repo_id or "").strip()
        # blank revision = latest
        self.revision = (revision or "").strip() or None
        self.token_provider = token_provider
        self._manifest = manifest
        self._temp_cache = {}
        self._repo_index = None

    def _repo_paths(self):
        # lookup_key -> the file's real path in the repo. Manifest paths can be
        # cased differently to what the repo ships (Agamidae_... vs agamidae_...)
        # and HF is case-sensitive, so grab the actual name. Listed once + cached.
        # Empty (offline/auth) just falls back to the manifest path.
        if self._repo_index is None:
            self._repo_index = {}
            try:
                from huggingface_hub import list_repo_files
            except ImportError:
                return self._repo_index
            try:
                token = self.token_provider() if self.token_provider else None
            except ValueError:
                token = None
            try:
                files = list_repo_files(
                    self.repo_id, repo_type="dataset", revision=self.revision, token=token)
            except Exception:
                return self._repo_index
            for f in files:
                if f.endswith(GLB_SUFFIX):
                    self._repo_index.setdefault(compute_lookup_key(f), f)
        return self._repo_index

    def _hf_path_for(self, mesh_name, row):
        # Prefer the real (correctly-cased) path from the repo listing; fall back to
        # the manifest's stored path when the listing is unavailable.
        real = self._repo_paths().get(compute_lookup_key(mesh_name))
        if real:
            return real
        return row["hf_path"] if row else None

    def manifest(self):
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self):
        # Nothing to resolve until the user enters a dataset repo.
        if not self.repo_id:
            return ReferenceManifest([])
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            return ReferenceManifest([])
        try:
            token = self.token_provider() if self.token_provider else None
        except ValueError:
            token = None
        try:
            files = list_repo_files(
                self.repo_id, repo_type="dataset", revision=self.revision, token=token)
        except Exception:
            return ReferenceManifest([])
        # 1. A curated manifest CSV shipped in the dataset - gives the full metadata.
        #    Matched by name so it can be manifest.csv, manifest_v44.csv, etc.
        manifest_file = _pick_manifest_file(files)
        if manifest_file:
            try:
                path = hf_hub_download(
                    self.repo_id, manifest_file, repo_type="dataset",
                    revision=self.revision, token=token)
                return ReferenceManifest.load(path)
            except Exception:
                pass
        # 2. No manifest CSV - derive the name->file map from the .glb listing.
        #    Metadata columns stay blank; only lookup_key + hf_path drive resolution.
        return ReferenceManifest.from_hf_paths(files)

    def availability(self, mesh_name):
        if not self.repo_id:
            return STATE_NO_REPO
        row = self.manifest().lookup(mesh_name)
        hf_path = self._hf_path_for(mesh_name, row)
        if hf_path is None:
            return STATE_MISSING
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return STATE_DOWNLOAD
        cached = try_to_load_from_cache(
            self.repo_id, hf_path, repo_type="dataset", revision=self.revision)
        return STATE_CACHED if isinstance(cached, str) else STATE_DOWNLOAD

    def resolve(self, mesh_name):
        if not self.repo_id:
            return Resolution("missing", None,
                              "Enter a HuggingFace dataset repo id to use the online library.",
                              STATE_NO_REPO)
        if mesh_name in self._temp_cache and os.path.isfile(self._temp_cache[mesh_name]):
            return Resolution("path", self._temp_cache[mesh_name], state=STATE_CACHED)

        row = self.manifest().lookup(mesh_name)
        hf_path = self._hf_path_for(mesh_name, row)
        if hf_path is None:
            return Resolution("missing", None,
                              "Missing from dataset: {}".format(mesh_name), STATE_MISSING)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return Resolution("missing", None,
                              "huggingface_hub is not installed.", STATE_MISSING)

        try:
            token = self.token_provider() if self.token_provider else None
        except ValueError as e:
            return Resolution("missing", None, str(e), STATE_AUTH)

        try:
            glb_path = hf_hub_download(
                self.repo_id, hf_path, repo_type="dataset",
                revision=self.revision, token=token)
        except Exception as e:
            category = map_download_error(e)
            if category == CATEGORY_AUTH:
                return Resolution("missing", None,
                                  "Auth required for {} - check your HuggingFace token.".format(mesh_name),
                                  STATE_AUTH)
            if category == CATEGORY_NETWORK:
                return Resolution("missing", None,
                                  "Network error fetching {} - check your network.".format(mesh_name),
                                  STATE_NETWORK)
            return Resolution("missing", None,
                              "Could not fetch {}".format(mesh_name), STATE_MISSING)

        # Decode (CATEGORY_DECODE): a bad/absent DracoPy wheel or corrupt .glb must not crash the slot.
        try:
            out_path = decode_glb_to_vtk(glb_path, compute_lookup_key(mesh_name))
        except Exception as e:
            return Resolution("missing", None,
                              "Could not decode {}: {}".format(mesh_name, e), STATE_MISSING)

        self._temp_cache[mesh_name] = out_path
        return Resolution("path", out_path, state=STATE_CACHED)


def select_backend(source, local_root=None, token_provider=None, manifest=None):
    # source: "local" | "huggingface"
    if source == "huggingface":
        return HuggingFaceBackend(token_provider=token_provider, manifest=manifest)
    return LocalFolderBackend(local_root)
