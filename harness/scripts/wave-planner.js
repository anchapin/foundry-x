#!/usr/bin/env node

const fs = require("fs");

const MAX_PER_WAVE = 3;

function readInput() {
  if (process.argv.length > 2) {
    return JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  }
  return JSON.parse(fs.readFileSync("/dev/stdin", "utf8"));
}

function extractFileRefs(text) {
  if (!text) return [];
  const files = new Set();

  const pathPatterns = [
    /[`"'[\s]([a-zA-Z0-9_./-]+\/(?:src|lib|test|tests|pkg|cmd|internal|osimflow|bin|docs|scripts|app|modules|components)\/[a-zA-Z0-9_./-]+\.[a-z]{2,4})[`"'[\s]/g,
    /[`"'[\s]([a-zA-Z0-9_.-]+\.[a-z]{2,4}):(\d+)/g,
    /[`"']([a-zA-Z0-9_/.-]+\.[a-z]{2,4})[`"']/g,
  ];

  for (const pat of pathPatterns) {
    let m;
    while ((m = pat.exec(text)) !== null) {
      const f = m[1];
      if (!f.includes("http") && !f.includes("://") && f.length > 3) {
        files.add(f);
      }
    }
  }

  return [...files];
}

function extractModuleRefs(text) {
  if (!text) return [];
  const modules = new Set();

  const patterns = [
    /import\s+.+\s+from\s+['"](\.?\.?\/[^'"]+)['"]/g,
    /from\s+([a-zA-Z0-9_.]+)\s+import/g,
    /require\(['"](\.?\.?\/[^'"]+)['"]\)/g,
    /use\s+([a-zA-Z0-9_:]+::[a-zA-Z0-9_:]+)/g,
  ];

  for (const pat of patterns) {
    let m;
    while ((m = pat.exec(text)) !== null) {
      modules.add(m[1]);
    }
  }

  return [...modules];
}

function analyzeIssue(issue) {
  const body = issue.body || "";
  const title = issue.title || "";
  const fullText = `${title}\n${body}`;

  const fileRefs = extractFileRefs(fullText);
  const moduleRefs = extractModuleRefs(fullText);

  const affectedFiles = [...new Set([...fileRefs, ...moduleRefs])];
  const hasKnownDeps = affectedFiles.length > 0;

  return {
    number: issue.number,
    title: title,
    labels: (issue.labels || []).map((l) =>
      typeof l === "string" ? l : l.name || ""
    ),
    affected_files: affectedFiles,
    has_known_deps: hasKnownDeps,
  };
}

function buildConflictGraph(analyzed) {
  const n = analyzed.length;
  const adj = Array.from({ length: n }, () => new Set());

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = analyzed[i];
      const b = analyzed[j];

      const sharesFiles = a.affected_files.some((f) =>
        b.affected_files.includes(f)
      );
      if (sharesFiles) {
        adj[i].add(j);
        adj[j].add(i);
      }
    }
  }

  return adj;
}

function graphColoring(adj, n, maxPerColor) {
  const colors = new Array(n).fill(-1);
  const colorCounts = [];

  for (let node = 0; node < n; node++) {
    const usedColors = new Set();
    for (const neighbor of adj[node]) {
      if (colors[neighbor] !== -1) {
        usedColors.add(colors[neighbor]);
      }
    }

    let assigned = -1;
    for (let c = 0; c < colorCounts.length; c++) {
      if (!usedColors.has(c) && colorCounts[c] < maxPerColor) {
        assigned = c;
        break;
      }
    }

    if (assigned === -1) {
      assigned = colorCounts.length;
      colorCounts.push(0);
    }

    colors[node] = assigned;
    colorCounts[assigned]++;
  }

  return colors;
}

function planWaves(issues) {
  if (!issues || issues.length === 0) {
    return { waves: [], total_issues: 0, total_waves: 0 };
  }

  const analyzed = issues.map(analyzeIssue);
  const adj = buildConflictGraph(analyzed);
  const colors = graphColoring(adj, analyzed.length, MAX_PER_WAVE);

  const maxWave = Math.max(...colors) + 1;
  const waves = [];

  for (let w = 0; w < maxWave; w++) {
    const waveIssues = analyzed.filter((_, i) => colors[i] === w);
    if (waveIssues.length > 0) {
      waves.push({
        wave: w + 1,
        issues: waveIssues,
      });
    }
  }

  return {
    total_issues: analyzed.length,
    total_waves: waves.length,
    waves,
  };
}

function main() {
  const input = readInput();
  const issues = Array.isArray(input) ? input : input.issues || [];
  const plan = planWaves(issues);
  console.log(JSON.stringify(plan, null, 2));
}

main();
