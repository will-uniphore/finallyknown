import { execFileSync } from "child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "fs";
import { homedir } from "os";
import { basename, dirname, extname, join, relative } from "path";
import { fileURLToPath } from "url";
import { getConfig } from "./config.js";
import { KnownDB } from "./db.js";
import { ingest } from "./ingest.js";

const HOME_DIR = homedir();
const DAY_MS = 24 * 60 * 60 * 1000;
const MIN_SCAN_BLOCK_CHARS = 500;
const MAX_GIT_REPOS_PER_SECTION = 15;
const MAX_BRANCHES_PER_REPO = 6;
const MAX_LANGUAGES_PER_REPO = 4;
const MAX_SHELL_COMMANDS = 500;
const MAX_COMMANDS_PER_CATEGORY = 25;
const MAX_FILE_GROUPS = 20;
const MAX_APPS_PER_CATEGORY = 20;
const MAX_DOCUMENT_GROUPS = 12;
const GIT_SKIP_NAMES = new Set([
  ".pnpm",
  ".pnpm-store",
  ".cache",
  ".cargo",
  ".rustup",
  ".npm",
  ".Trash",
  "node_modules",
  "vendor",
  "Library",
  "Movies",
  "Music",
  "Pictures",
]);
const FILE_SKIP_NAMES = new Set([
  "Applications",
  "Library",
  "Movies",
  "Music",
  "Pictures",
  "Public",
  "node_modules",
  "vendor",
  "dist",
  "build",
  "coverage",
  "target",
  ".git",
]);
const SHELL_SECRET_PATTERN = /(password|token|secret|key|sk-|ghp_)/i;
const RECENT_DOCUMENT_FALLBACK_DIRS = ["Documents", "Desktop", "Downloads"];

type ScanSource = "git" | "shell" | "files" | "apps" | "calendar";

type ScanOptions = {
  sources: Set<ScanSource>;
  dryRun: boolean;
};

type ScanBlock = {
  source: ScanSource;
  sessionId: string;
  text: string;
};

type RepoSummary = {
  path: string;
  lastCommitAt: Date;
  commitsLast30Days: number;
  branches: string[];
  languages: string[];
};

type DirectorySummary = {
  path: string;
  count: number;
  latestMtimeMs: number;
  extensions: Map<string, number>;
};

type CalendarSummary = {
  totalEvents: number;
  busyDays: Map<string, number>;
  meetingTypes: Map<string, number>;
};

function splitLines(value: string | null): string[] {
  return (value ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function runCommand(command: string, args: string[], cwd?: string): string | null {
  try {
    return execFileSync(command, args, {
      cwd,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 10_000,
      maxBuffer: 16 * 1024 * 1024,
    }).trim();
  } catch (error) {
    if (
      error &&
      typeof error === "object" &&
      "stdout" in error &&
      typeof (error as { stdout?: unknown }).stdout === "string"
    ) {
      return (error as { stdout: string }).stdout.trim();
    }
    return null;
  }
}

function formatHomePath(inputPath: string): string {
  const relativePath = relative(HOME_DIR, inputPath);
  if (!relativePath || relativePath === "") {
    return "~";
  }
  if (relativePath.startsWith("..")) {
    return inputPath;
  }
  return `~/${relativePath}`;
}

function formatAge(timestamp: Date): string {
  const diffMs = Math.max(0, Date.now() - timestamp.getTime());
  const hours = diffMs / (60 * 60 * 1000);
  if (hours < 1) {
    return "just now";
  }
  if (hours < 24) {
    return `${Math.round(hours)}h ago`;
  }

  const days = diffMs / DAY_MS;
  if (days < 30) {
    return `${Math.round(days)} days ago`;
  }
  if (days < 365) {
    return `${Math.round(days / 30)} months ago`;
  }
  return `${Math.round(days / 365)} years ago`;
}

function formatPathLabel(inputPath: string): string {
  return basename(inputPath) || formatHomePath(inputPath);
}

function normalizeExtension(inputPath: string): string {
  const extension = extname(inputPath).toLowerCase();
  return extension.startsWith(".") ? extension : "";
}

function extensionDisplay(extension: string): string {
  return extension.startsWith(".") ? extension.slice(1) : extension || "(none)";
}

function topKeys(map: Map<string, number>, limit: number): string[] {
  return [...map.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit)
    .map(([key]) => key);
}

function countBy<T>(items: T[], keyFn: (item: T) => string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyFn(item);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function inferLanguageFromExtension(extension: string): string | null {
  const mapping: Record<string, string> = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".deno": "Deno",
    ".elm": "Elm",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".go": "Go",
    ".h": "C/C++",
    ".hpp": "C++",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript/React",
    ".kt": "Kotlin",
    ".lua": "Lua",
    ".m": "Objective-C",
    ".md": "Markdown",
    ".mm": "Objective-C++",
    ".php": "PHP",
    ".pl": "Perl",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".scss": "SCSS",
    ".sh": "Shell",
    ".sql": "SQL",
    ".swift": "Swift",
    ".svelte": "Svelte",
    ".ts": "TypeScript",
    ".tsx": "TypeScript/React",
    ".vue": "Vue",
    ".zsh": "Shell",
  };
  return mapping[extension] ?? null;
}

function summarizeLanguagesFromFiles(files: string[]): string[] {
  const counts = new Map<string, number>();
  for (const file of files) {
    const language = inferLanguageFromExtension(normalizeExtension(file));
    if (!language) {
      continue;
    }
    counts.set(language, (counts.get(language) ?? 0) + 1);
  }

  return topKeys(counts, MAX_LANGUAGES_PER_REPO);
}

function buildScanBlock(source: ScanSource, title: string, lines: string[]): ScanBlock | null {
  const filteredLines = lines.map((line) => line.trimEnd()).filter((line) => line.length > 0);
  if (filteredLines.length === 0) {
    return null;
  }

  const text = [
    `=== ${title} ===`,
    "This scan was generated locally from the user's machine without OAuth, remote APIs, or internet access.",
    "Treat it as behavioral evidence about recurring work patterns, tool choices, schedules, and environments rather than as a list of one-off tasks.",
    ...filteredLines,
  ].join("\n");

  return {
    source,
    sessionId: `local-scan-${source}`,
    text,
  };
}

function parseScanArgs(args: string[]): ScanOptions {
  const knownFlags = new Set(["--git", "--shell", "--files", "--apps", "--calendar", "--dry-run"]);
  const sources = new Set<ScanSource>();
  let dryRun = false;

  for (const arg of args) {
    if (!knownFlags.has(arg)) {
      throw new Error(`Usage error: known scan [--git] [--shell] [--files] [--apps] [--calendar] [--dry-run]`);
    }

    if (arg === "--dry-run") {
      dryRun = true;
      continue;
    }

    if (arg === "--git") sources.add("git");
    if (arg === "--shell") sources.add("shell");
    if (arg === "--files") sources.add("files");
    if (arg === "--apps") sources.add("apps");
    if (arg === "--calendar") sources.add("calendar");
  }

  if (sources.size === 0) {
    sources.add("git");
    sources.add("shell");
    sources.add("files");
    sources.add("apps");
    sources.add("calendar");
  }

  return { sources, dryRun };
}

function findPathsWithFind(args: string[]): string[] {
  return splitLines(runCommand("find", args));
}

function listHomeRoots(skipNames: Set<string>): string[] {
  const roots: string[] = [];

  try {
    for (const entry of readdirSync(HOME_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }
      if (entry.name.startsWith(".") || skipNames.has(entry.name)) {
        continue;
      }
      roots.push(join(HOME_DIR, entry.name));
    }
  } catch {
    return [];
  }

  return roots;
}

function findGitRepositories(): string[] {
  const pruneNames = [...GIT_SKIP_NAMES];
  const roots = listHomeRoots(GIT_SKIP_NAMES);
  const repos = new Set<string>();

  for (const root of roots) {
    const args = [
      root,
      "(",
      ...pruneNames.flatMap((name, index) => (index === 0 ? ["-name", name] : ["-o", "-name", name])),
      ")",
      "-prune",
      "-o",
      "-name",
      ".git",
      "-type",
      "d",
      "-print",
    ];

    for (const gitDir of findPathsWithFind(args)) {
      repos.add(dirname(gitDir));
    }
  }

  return [...repos];
}

function collectRepoSummary(repoPath: string): RepoSummary | null {
  const lastCommitRaw = runCommand("git", ["-C", repoPath, "log", "-1", "--format=%cI"]);
  if (!lastCommitRaw) {
    return null;
  }

  const lastCommitAt = new Date(lastCommitRaw);
  if (Number.isNaN(lastCommitAt.getTime())) {
    return null;
  }

  const sinceIso = new Date(Date.now() - 30 * DAY_MS).toISOString();
  const commitsLast30Days = Number.parseInt(
    runCommand("git", ["-C", repoPath, "rev-list", "--count", `--since=${sinceIso}`, "HEAD"]) ?? "0",
    10,
  );
  const branches = splitLines(runCommand("git", ["-C", repoPath, "for-each-ref", "--format=%(refname:short)", "refs/heads"])).slice(
    0,
    MAX_BRANCHES_PER_REPO,
  );
  const trackedFiles = splitLines(runCommand("git", ["-C", repoPath, "ls-files"]));
  const languages = summarizeLanguagesFromFiles(trackedFiles);

  return {
    path: repoPath,
    lastCommitAt,
    commitsLast30Days: Number.isFinite(commitsLast30Days) ? commitsLast30Days : 0,
    branches,
    languages,
  };
}

async function scanGitActivity(): Promise<ScanBlock | null> {
  const repoPaths = findGitRepositories();
  const repos = repoPaths
    .map((repoPath) => collectRepoSummary(repoPath))
    .filter((repo): repo is RepoSummary => Boolean(repo))
    .sort((left, right) => right.lastCommitAt.getTime() - left.lastCommitAt.getTime());

  if (repos.length === 0) {
    return null;
  }

  const active = repos.filter((repo) => Date.now() - repo.lastCommitAt.getTime() <= 7 * DAY_MS).slice(0, MAX_GIT_REPOS_PER_SECTION);
  const recent = repos
    .filter((repo) => Date.now() - repo.lastCommitAt.getTime() > 7 * DAY_MS && Date.now() - repo.lastCommitAt.getTime() <= 30 * DAY_MS)
    .slice(0, MAX_GIT_REPOS_PER_SECTION);
  const dormant = repos
    .filter((repo) => Date.now() - repo.lastCommitAt.getTime() > 30 * DAY_MS && Date.now() - repo.lastCommitAt.getTime() <= 90 * DAY_MS)
    .slice(0, MAX_GIT_REPOS_PER_SECTION);

  const lines: string[] = [
    `Discovered ${repos.length} Git repositories under ${formatHomePath(HOME_DIR)}.`,
    "Languages are inferred from tracked file extensions and branch names are local branches only.",
    "",
    "Active repos (last commit within 7 days):",
  ];

  if (active.length === 0) {
    lines.push("- none");
  } else {
    for (const repo of active) {
      lines.push(
        `- ${formatHomePath(repo.path)}: ${repo.commitsLast30Days} commits last 30 days, ${
          repo.languages.join(", ") || "language mix unclear"
        }, last commit ${formatAge(repo.lastCommitAt)}`,
      );
    }
  }

  lines.push("", "Recently active repos (7-30 days):");
  if (recent.length === 0) {
    lines.push("- none");
  } else {
    for (const repo of recent) {
      lines.push(
        `- ${formatHomePath(repo.path)}: ${repo.commitsLast30Days} commits last 30 days, ${
          repo.languages.join(", ") || "language mix unclear"
        }, last commit ${formatAge(repo.lastCommitAt)}`,
      );
    }
  }

  lines.push("", "Dormant repos (30-90 days):");
  if (dormant.length === 0) {
    lines.push("- none");
  } else {
    for (const repo of dormant) {
      lines.push(
        `- ${formatHomePath(repo.path)}: ${repo.commitsLast30Days} commits last 30 days, ${
          repo.languages.join(", ") || "language mix unclear"
        }, last commit ${formatAge(repo.lastCommitAt)}`,
      );
    }
  }

  lines.push("", "Branch names suggest:");
  for (const repo of repos.slice(0, 10)) {
    if (repo.branches.length === 0) {
      continue;
    }
    lines.push(`- ${formatPathLabel(repo.path)}: ${repo.branches.join(", ")}`);
  }

  return buildScanBlock("git", "GIT ACTIVITY", lines);
}

function readShellHistoryFile(): { path: string; lines: string[] } | null {
  const candidates = [join(HOME_DIR, ".zsh_history"), join(HOME_DIR, ".bash_history")];
  for (const historyPath of candidates) {
    if (!existsSync(historyPath)) {
      continue;
    }

    try {
      return {
        path: historyPath,
        lines: readFileSync(historyPath, "utf8").split(/\r?\n/),
      };
    } catch {
      continue;
    }
  }

  return null;
}

function parseHistoryCommand(rawLine: string): string {
  const zshSeparator = rawLine.indexOf(";");
  if (rawLine.startsWith(": ") && zshSeparator !== -1) {
    return rawLine.slice(zshSeparator + 1).trim();
  }
  return rawLine.trim();
}

function categorizeCommand(command: string): string | null {
  if (/^git\b/i.test(command)) {
    return "Git";
  }
  if (/^(npm|pnpm|yarn|bun)\b/i.test(command)) {
    return "Package managers";
  }
  if (/^(docker|docker-compose|kubectl|helm)\b/i.test(command)) {
    return "Containers and infra";
  }
  if (/^(node|tsx|ts-node|tsc|vite|vitest|jest|python|python3|pip|pipx|uv|pytest|go|cargo|rustc|ruby|bundle|rails|deno|java|javac)\b/i.test(command)) {
    return "Languages and runtimes";
  }
  return null;
}

async function scanShellHistory(): Promise<ScanBlock | null> {
  const history = readShellHistoryFile();
  if (!history) {
    return null;
  }

  const seen = new Set<string>();
  const categorized = new Map<string, string[]>();

  for (let index = history.lines.length - 1; index >= 0; index -= 1) {
    const parsed = parseHistoryCommand(history.lines[index]);
    const normalized = parsed.replace(/\s+/g, " ").trim();
    if (!normalized || seen.has(normalized.toLowerCase())) {
      continue;
    }
    if (SHELL_SECRET_PATTERN.test(normalized)) {
      continue;
    }

    const category = categorizeCommand(normalized);
    if (!category) {
      continue;
    }

    seen.add(normalized.toLowerCase());
    const bucket = categorized.get(category) ?? [];
    bucket.push(normalized);
    categorized.set(category, bucket);

    if (seen.size >= MAX_SHELL_COMMANDS) {
      break;
    }
  }

  if (seen.size === 0) {
    return null;
  }

  const lines: string[] = [
    `Source history file: ${formatHomePath(history.path)}.`,
    `Sampled the last ${seen.size} unique safe commands after removing duplicates and commands containing password, token, secret, key, sk-, or ghp_.`,
  ];

  for (const category of ["Git", "Package managers", "Containers and infra", "Languages and runtimes"]) {
    const commands = categorized.get(category);
    if (!commands || commands.length === 0) {
      continue;
    }

    lines.push("", `${category}:`);
    for (const command of commands.slice(0, MAX_COMMANDS_PER_CATEGORY)) {
      lines.push(`- ${command}`);
    }
  }

  return buildScanBlock("shell", "SHELL HISTORY", lines);
}

function findRecentFiles(): string[] {
  const roots = listHomeRoots(FILE_SKIP_NAMES);
  const recentFiles = new Set<string>();
  const pruneTerms = ["-name", ".*", ...[...FILE_SKIP_NAMES].flatMap((name) => ["-o", "-name", name])];

  for (const root of roots) {
    const args = [
      root,
      "(",
      "-type",
      "d",
      "(",
      ...pruneTerms,
      ")",
      "-prune",
      ")",
      "-o",
      "-type",
      "f",
      "-mtime",
      "-7",
      "-print",
    ];

    for (const filePath of findPathsWithFind(args)) {
      recentFiles.add(filePath);
    }
  }

  return [...recentFiles];
}

function summarizeDirectoryActivity(paths: string[], limit: number): DirectorySummary[] {
  const summaries = new Map<string, DirectorySummary>();

  for (const filePath of paths) {
    let stats;
    try {
      stats = statSync(filePath);
    } catch {
      continue;
    }

    const parentPath = dirname(filePath);
    const extension = normalizeExtension(filePath);
    const existing = summaries.get(parentPath) ?? {
      path: parentPath,
      count: 0,
      latestMtimeMs: 0,
      extensions: new Map<string, number>(),
    };

    existing.count += 1;
    existing.latestMtimeMs = Math.max(existing.latestMtimeMs, stats.mtimeMs);
    if (extension) {
      existing.extensions.set(extension, (existing.extensions.get(extension) ?? 0) + 1);
    }
    summaries.set(parentPath, existing);
  }

  return [...summaries.values()]
    .sort((left, right) => right.count - left.count || right.latestMtimeMs - left.latestMtimeMs)
    .slice(0, limit);
}

function summarizeExtensionList(extensions: Map<string, number>, limit: number): string {
  const summary = topKeys(extensions, limit).map(extensionDisplay);
  return summary.length > 0 ? summary.join(", ") : "no clear extension pattern";
}

function extractRecentDocumentPathsFromSfl3(): string[] {
  const recentDocumentsPath = join(
    HOME_DIR,
    "Library",
    "Application Support",
    "com.apple.sharedfilelist",
    "com.apple.LSSharedFileList.RecentDocuments.sfl3",
  );
  if (!existsSync(recentDocumentsPath)) {
    return [];
  }

  const jsonText = runCommand("plutil", ["-convert", "json", "-o", "-", recentDocumentsPath]);
  if (!jsonText) {
    return [];
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch {
    return [];
  }

  const urls = new Set<string>();
  const stack: unknown[] = [parsed];
  while (stack.length > 0) {
    const current = stack.pop();
    if (typeof current === "string" && current.startsWith("file://")) {
      urls.add(current);
      continue;
    }
    if (Array.isArray(current)) {
      stack.push(...current);
      continue;
    }
    if (current && typeof current === "object") {
      stack.push(...Object.values(current as Record<string, unknown>));
    }
  }

  const paths: string[] = [];
  for (const url of urls) {
    try {
      paths.push(fileURLToPath(url));
    } catch {
      continue;
    }
  }

  return paths;
}

function findFallbackRecentDocumentPaths(): string[] {
  const roots = RECENT_DOCUMENT_FALLBACK_DIRS.map((dir) => join(HOME_DIR, dir)).filter((dir) => existsSync(dir));
  const allPaths = new Set<string>();

  for (const root of roots) {
    const args = [root, "-type", "f", "-mtime", "-14", "-print"];
    for (const filePath of findPathsWithFind(args)) {
      allPaths.add(filePath);
    }
  }

  return [...allPaths];
}

async function scanFileSystemActivity(): Promise<ScanBlock | null> {
  const recentFiles = findRecentFiles();
  const recentFileGroups = summarizeDirectoryActivity(recentFiles, MAX_FILE_GROUPS);
  const recentDocumentPaths = extractRecentDocumentPathsFromSfl3();
  const recentDocuments = recentDocumentPaths.length > 0 ? recentDocumentPaths : findFallbackRecentDocumentPaths();
  const recentDocumentGroups = summarizeDirectoryActivity(recentDocuments, MAX_DOCUMENT_GROUPS);

  if (recentFileGroups.length === 0 && recentDocumentGroups.length === 0) {
    return null;
  }

  const lines: string[] = [
    "Recently modified files over the last 7 days are grouped by parent directory to show active projects and working surfaces.",
    "Recent documents are summarized by directory and file type only, without listing specific filenames.",
    "",
    "Recent file activity:",
  ];

  if (recentFileGroups.length === 0) {
    lines.push("- none");
  } else {
    for (const summary of recentFileGroups) {
      lines.push(
        `- ${formatHomePath(summary.path)}: ${summary.count} files modified, last touched ${formatAge(
          new Date(summary.latestMtimeMs),
        )}, extensions: ${summarizeExtensionList(summary.extensions, 5)}`,
      );
    }
  }

  lines.push("", "Recent document patterns:");
  if (recentDocumentGroups.length === 0) {
    lines.push("- none");
  } else {
    for (const summary of recentDocumentGroups) {
      lines.push(
        `- ${formatHomePath(summary.path)}: ${summary.count} recent documents, latest activity ${formatAge(
          new Date(summary.latestMtimeMs),
        )}, file types: ${summarizeExtensionList(summary.extensions, 4)}`,
      );
    }
  }

  return buildScanBlock("files", "FILE SYSTEM ACTIVITY", lines);
}

function categorizeApp(appName: string): string | null {
  const normalized = appName.toLowerCase();

  if (
    /(cursor|visual studio code|code|xcode|terminal|iterm|warp|docker|postman|insomnia|tableplus|db browser|github desktop|gitkraken|sourcetree|claude|openclaw|android studio)/.test(
      normalized,
    )
  ) {
    return "Dev tools";
  }
  if (/(slack|zoom|discord|teams|mail|messages|outlook|telegram|whatsapp|signal)/.test(normalized)) {
    return "Communication";
  }
  if (/(figma|sketch|photoshop|illustrator|pixelmator|canva|framer|principle)/.test(normalized)) {
    return "Design";
  }
  if (/(notion|obsidian|calendar|reminders|notes|todoist|things|raycast|1password|excel|word|powerpoint)/.test(normalized)) {
    return "Productivity";
  }

  return null;
}

async function scanInstalledApps(): Promise<ScanBlock | null> {
  const appRoots = ["/Applications", join(HOME_DIR, "Applications")];
  const categorized = new Map<string, string[]>();

  for (const root of appRoots) {
    if (!existsSync(root)) {
      continue;
    }

    let entries: string[] = [];
    try {
      entries = readdirSync(root);
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (!entry.endsWith(".app")) {
        continue;
      }

      const appName = entry.replace(/\.app$/i, "");
      const category = categorizeApp(appName);
      if (!category) {
        continue;
      }

      const bucket = categorized.get(category) ?? [];
      if (!bucket.includes(appName)) {
        bucket.push(appName);
      }
      categorized.set(category, bucket);
    }
  }

  if (categorized.size === 0) {
    return null;
  }

  const lines: string[] = [
    "Applications were listed from /Applications and ~/Applications, then grouped into coarse categories to reveal the local tool environment.",
  ];

  for (const category of ["Dev tools", "Productivity", "Communication", "Design"]) {
    const apps = (categorized.get(category) ?? []).sort((left, right) => left.localeCompare(right)).slice(0, MAX_APPS_PER_CATEGORY);
    if (apps.length === 0) {
      continue;
    }

    lines.push("", `${category}:`);
    for (const app of apps) {
      lines.push(`- ${app}`);
    }
  }

  return buildScanBlock("apps", "INSTALLED APPS", lines);
}

function unfoldIcs(text: string): string {
  return text.replace(/\r?\n[ \t]/g, "");
}

function parseIcsDate(value: string): Date | null {
  const trimmed = value.trim();
  const dateOnlyMatch = /^(\d{4})(\d{2})(\d{2})$/.exec(trimmed);
  if (dateOnlyMatch) {
    const [, year, month, day] = dateOnlyMatch;
    return new Date(Number(year), Number(month) - 1, Number(day));
  }

  const dateTimeUtcMatch = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(trimmed);
  if (dateTimeUtcMatch) {
    const [, year, month, day, hour, minute, second] = dateTimeUtcMatch;
    return new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)));
  }

  const dateTimeLocalMatch = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/.exec(trimmed);
  if (dateTimeLocalMatch) {
    const [, year, month, day, hour, minute, second] = dateTimeLocalMatch;
    return new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second));
  }

  return null;
}

function meetingTypeFromSummary(summary: string): string | null {
  const normalized = summary.toLowerCase();
  if (/\b1[: ]?1\b|one[- ]on[- ]one/.test(normalized)) {
    return "1:1";
  }
  if (/\bstandup\b|\bdaily\b/.test(normalized)) {
    return "Standup";
  }
  if (/\breview\b|\bdemo\b|\bretro\b|\bretrospective\b/.test(normalized)) {
    return "Review/demo";
  }
  if (/\bplanning\b|\bplan\b/.test(normalized)) {
    return "Planning";
  }
  if (/\bsync\b/.test(normalized)) {
    return "Sync";
  }
  return null;
}

function summarizeCalendarFile(filePath: string, summary: CalendarSummary, cutoff: number) {
  let raw = "";
  try {
    raw = readFileSync(filePath, "utf8");
  } catch {
    return;
  }

  const unfolded = unfoldIcs(raw);
  const events = unfolded.split("BEGIN:VEVENT").slice(1);
  for (const event of events) {
    const lines = event.split(/\r?\n/);
    let startAt: Date | null = null;
    let summaryText = "";

    for (const line of lines) {
      const separatorIndex = line.indexOf(":");
      if (separatorIndex === -1) {
        continue;
      }

      const rawKey = line.slice(0, separatorIndex);
      const value = line.slice(separatorIndex + 1);
      const key = rawKey.split(";")[0].toUpperCase();

      if (key === "DTSTART") {
        startAt = parseIcsDate(value);
      }
      if (key === "SUMMARY") {
        summaryText = value;
      }
    }

    if (!startAt || Number.isNaN(startAt.getTime()) || startAt.getTime() < cutoff) {
      continue;
    }

    summary.totalEvents += 1;
    const dayName = startAt.toLocaleDateString("en-US", { weekday: "long" });
    summary.busyDays.set(dayName, (summary.busyDays.get(dayName) ?? 0) + 1);

    const meetingType = meetingTypeFromSummary(summaryText);
    if (meetingType) {
      summary.meetingTypes.set(meetingType, (summary.meetingTypes.get(meetingType) ?? 0) + 1);
    }
  }
}

async function scanCalendarPatterns(): Promise<ScanBlock | null> {
  const calendarRoot = join(HOME_DIR, "Library", "Calendars");
  if (!existsSync(calendarRoot)) {
    return null;
  }

  const calendarFiles = findPathsWithFind([calendarRoot, "-name", "*.ics", "-type", "f"]);
  if (calendarFiles.length === 0) {
    return null;
  }

  const cutoff = Date.now() - 90 * DAY_MS;
  const summary: CalendarSummary = {
    totalEvents: 0,
    busyDays: new Map<string, number>(),
    meetingTypes: new Map<string, number>(),
  };

  for (const filePath of calendarFiles) {
    summarizeCalendarFile(filePath, summary, cutoff);
  }

  if (summary.totalEvents === 0) {
    return null;
  }

  const busyDays = [...summary.busyDays.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3)
    .map(([day, count]) => `${day} (${count})`);
  const meetingTypes = [...summary.meetingTypes.entries()]
    .sort((left, right) => right[1] - left[1])
    .map(([type, count]) => `${type} (${count})`);

  const eventsPerWeek = (summary.totalEvents / (90 / 7)).toFixed(1);
  const lines: string[] = [
    `Calendar patterns are based on local .ics files from ${formatHomePath(calendarRoot)} and only summarize aggregate patterns over the last 90 days.`,
    `Parsed ${calendarFiles.length} calendar files and ${summary.totalEvents} events, which is about ${eventsPerWeek} events per week.`,
    `Busiest days of the week: ${busyDays.join(", ") || "not enough data"}.`,
    `Meeting types detected from summaries only: ${meetingTypes.join(", ") || "no labeled meeting patterns detected"}.`,
    "Event titles, attendee details, and raw calendar contents were intentionally excluded from the output.",
  ];

  return buildScanBlock("calendar", "CALENDAR PATTERNS", lines);
}

async function collectBlocks(options: ScanOptions): Promise<ScanBlock[]> {
  const blocks: ScanBlock[] = [];

  if (options.sources.has("git")) {
    const block = await scanGitActivity();
    if (block) blocks.push(block);
  }
  if (options.sources.has("shell")) {
    const block = await scanShellHistory();
    if (block) blocks.push(block);
  }
  if (options.sources.has("files")) {
    const block = await scanFileSystemActivity();
    if (block) blocks.push(block);
  }
  if (options.sources.has("apps")) {
    const block = await scanInstalledApps();
    if (block) blocks.push(block);
  }
  if (options.sources.has("calendar")) {
    const block = await scanCalendarPatterns();
    if (block) blocks.push(block);
  }

  return blocks;
}

function printDryRun(blocks: ScanBlock[]) {
  for (const block of blocks) {
    console.log(`--- ${block.sessionId} (${block.text.length} chars) ---`);
    console.log(block.text);
    console.log("");
  }
}

export async function runScanCli(args: string[]) {
  const options = parseScanArgs(args);
  const blocks = await collectBlocks(options);

  if (blocks.length === 0) {
    console.log("No local scan data found for the selected sources.");
    return;
  }

  if (options.dryRun) {
    printDryRun(blocks);
    return;
  }

  const config = getConfig();
  const db = new KnownDB(config.dbPath);

  try {
    let totalNodesCreated = 0;
    let totalEdgesCreated = 0;

    for (const block of blocks) {
      if (block.text.length < MIN_SCAN_BLOCK_CHARS) {
        console.log(`Skipping ${block.source}: scan block too short (${block.text.length} chars).`);
        continue;
      }

      const result = await ingest(db, block.text, config, block.sessionId);
      totalNodesCreated += result.nodesCreated;
      totalEdgesCreated += result.edgesCreated;
      console.log(
        `Scanned ${block.source}: ${result.nodesCreated} nodes created, ${result.edgesCreated} edges created (${block.text.length} chars).`,
      );
    }

    console.log(`Local scan complete: ${totalNodesCreated} nodes created, ${totalEdgesCreated} edges created.`);
  } finally {
    db.close();
  }
}
