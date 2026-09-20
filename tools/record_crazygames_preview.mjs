#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";


function parseArgs(argv) {
  const result = {
    url: "",
    orientation: "landscape",
    output: "",
    artifactDir: "",
    headed: false,
    softwareRendering: false,
    cpuThrottle: 1,
    viewportWidth: 0,
    viewportHeight: 0,
    codec: "vp9",
  };
  for (let index = 2; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];
    if (argument === "--url" && value) {
      result.url = value;
      index += 1;
    } else if (argument === "--orientation" && value) {
      result.orientation = value;
      index += 1;
    } else if (argument === "--output" && value) {
      result.output = value;
      index += 1;
    } else if (argument === "--artifact-dir" && value) {
      result.artifactDir = value;
      index += 1;
    } else if (argument === "--headed") {
      result.headed = true;
    } else if (argument === "--software-rendering") {
      result.softwareRendering = true;
    } else if (argument === "--cpu-throttle" && value) {
      result.cpuThrottle = Number.parseFloat(value);
      index += 1;
    } else if (argument === "--viewport-width" && value) {
      result.viewportWidth = Number.parseInt(value, 10);
      index += 1;
    } else if (argument === "--viewport-height" && value) {
      result.viewportHeight = Number.parseInt(value, 10);
      index += 1;
    } else if (argument === "--codec" && value) {
      result.codec = value;
      index += 1;
    }
  }
  if (!result.url || !result.output || !result.artifactDir) {
    throw new Error("--url, --output, and --artifact-dir are required");
  }
  if (!["landscape", "portrait"].includes(result.orientation)) {
    throw new Error("--orientation must be landscape or portrait");
  }
  if (!Number.isFinite(result.cpuThrottle) || result.cpuThrottle < 1) {
    throw new Error("--cpu-throttle must be a number greater than or equal to 1");
  }
  if (
    (result.viewportWidth === 0) !== (result.viewportHeight === 0)
    || result.viewportWidth < 0
    || result.viewportHeight < 0
  ) {
    throw new Error("--viewport-width and --viewport-height must be set together");
  }
  if (!["h264", "vp8", "vp9"].includes(result.codec)) {
    throw new Error("--codec must be h264, vp8, or vp9");
  }
  return result;
}


async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    const codexHome = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");
    const modulePath = path.join(
      codexHome,
      "skills",
      "develop-web-game",
      "node_modules",
      "playwright",
      "index.mjs",
    );
    if (!fs.existsSync(modulePath)) {
      throw new Error(`Playwright was not found at ${modulePath}`);
    }
    return import(pathToFileURL(modulePath).href);
  }
}


function readGameState(rawState) {
  if (typeof rawState !== "string" || rawState.length === 0) {
    return {};
  }
  try {
    return JSON.parse(rawState);
  } catch {
    return {};
  }
}


async function hold(page, key, milliseconds) {
  await page.keyboard.down(key);
  await page.waitForTimeout(milliseconds);
  await page.keyboard.up(key);
}


async function placeBubble(page) {
  await page.keyboard.press("Space");
  await page.waitForTimeout(120);
}


async function playPreviewRoute(page) {
  await page.waitForTimeout(600);
  await hold(page, "KeyD", 900);
  await placeBubble(page);
  await hold(page, "KeyW", 700);
  await placeBubble(page);
  await hold(page, "KeyD", 850);
  await hold(page, "KeyS", 650);
  await placeBubble(page);
  await hold(page, "KeyW", 1100);
  await hold(page, "KeyD", 800);
  await placeBubble(page);
  await hold(page, "KeyA", 1100);
  await placeBubble(page);
  await hold(page, "KeyS", 750);
  await hold(page, "KeyD", 900);
  await placeBubble(page);
  await hold(page, "KeyW", 1100);
  await hold(page, "KeyA", 950);
  await placeBubble(page);
  await hold(page, "KeyS", 800);
  await hold(page, "KeyD", 900);
  await placeBubble(page);
  await hold(page, "KeyA", 900);
  await hold(page, "KeyW", 900);
  await placeBubble(page);
  await hold(page, "KeyD", 700);
  await hold(page, "KeyS", 700);
  await page.waitForTimeout(700);
}


async function startCanvasRecording(
  page,
  framesPerSecond = 60,
  preferredCodec = "h264",
) {
  return page.evaluate(({ fps, codec }) => {
    const canvas = [...document.querySelectorAll("canvas")]
      .sort((left, right) => (
        (right.width * right.height) - (left.width * left.height)
      ))[0];
    if (!canvas) {
      throw new Error("No game canvas was found");
    }
    if (typeof canvas.captureStream !== "function") {
      throw new Error("canvas.captureStream is unavailable");
    }
    const mimeCandidates = {
      h264: [
        "video/mp4;codecs=avc1.64002A",
        "video/mp4;codecs=avc1.42E01E",
        "video/webm;codecs=h264",
      ],
      vp8: ["video/webm;codecs=vp8"],
      vp9: ["video/webm;codecs=vp9"],
    };
    const mimeType = [...mimeCandidates[codec], "video/webm"]
      .find((candidate) => MediaRecorder.isTypeSupported(candidate));
    if (!mimeType) {
      throw new Error("No supported MediaRecorder video codec was found");
    }
    const stream = canvas.captureStream(fps);
    const chunks = [];
    const recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 12_000_000,
    });
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.start(250);
    window.__bnbCanvasCapture = {
      chunks,
      mimeType,
      recorder,
      stream,
      width: canvas.width,
      height: canvas.height,
    };
    return {
      mime_type: mimeType,
      requested_fps: fps,
      source_width: canvas.width,
      source_height: canvas.height,
    };
  }, { fps: framesPerSecond, codec: preferredCodec });
}


async function stopCanvasRecording(page, outputPath) {
  const downloadPromise = page.waitForEvent("download", { timeout: 30000 });
  const resultPromise = page.evaluate(async (downloadName) => {
    const capture = window.__bnbCanvasCapture;
    if (!capture) {
      throw new Error("Canvas recording was not started");
    }
    await new Promise((resolve) => {
      capture.recorder.addEventListener("stop", resolve, { once: true });
      capture.recorder.stop();
    });
    for (const track of capture.stream.getTracks()) track.stop();
    const blob = new Blob(capture.chunks, { type: capture.mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = downloadName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    return {
      bytes: blob.size,
      mime_type: capture.mimeType,
    };
  }, path.basename(outputPath));
  const [download, result] = await Promise.all([
    downloadPromise,
    resultPromise,
  ]);
  await download.saveAs(path.resolve(outputPath));
  return result;
}


async function startPerformanceSampling(page) {
  await page.evaluate(() => {
    window.__bnbPerformanceSamples = [];
    window.__bnbPerformanceSampler = window.setInterval(() => {
      try {
        const state = JSON.parse(window.render_game_to_text());
        if (state.performance) {
          window.__bnbPerformanceSamples.push({
            captured_at_ms: performance.now(),
            ...state.performance,
          });
        }
      } catch {
        // A missing sample during a state transition is not a browser error.
      }
    }, 250);
  });
}


async function stopPerformanceSampling(page) {
  const samples = await page.evaluate(() => {
    window.clearInterval(window.__bnbPerformanceSampler);
    return window.__bnbPerformanceSamples || [];
  });
  if (samples.length === 0) return { samples: [] };
  const summarize = (values) => {
    const ordered = [...values].sort((left, right) => left - right);
    return {
      min: ordered[0],
      average: Number((
        ordered.reduce((total, value) => total + value, 0) / ordered.length
      ).toFixed(2)),
      p95: ordered[Math.min(
        ordered.length - 1,
        Math.ceil(ordered.length * 0.95) - 1,
      )],
      max: ordered[ordered.length - 1],
    };
  };
  const summarizeSampleSet = (sampleSet) => {
    const numeric = (key) => sampleSet
      .map((sample) => Number(sample[key]))
      .filter(Number.isFinite);
    return {
      sample_count: sampleSet.length,
      fps: summarize(numeric("fps")),
      process_ms: summarize(numeric("process_ms")),
      physics_ms: summarize(numeric("physics_ms")),
      draw_calls: summarize(numeric("draw_calls")),
      objects: summarize(numeric("objects")),
    };
  };
  const warmupMilliseconds = 2000;
  const steadySamples = samples.filter((sample) => (
    sample.captured_at_ms
      >= samples[0].captured_at_ms + warmupMilliseconds
  ));
  return {
    warmup_excluded_ms: warmupMilliseconds,
    all_samples: summarizeSampleSet(samples),
    steady_state: summarizeSampleSet(steadySamples),
    samples,
  };
}


async function main() {
  const args = parseArgs(process.argv);
  const { chromium } = await loadPlaywright();
  const viewport = args.viewportWidth > 0
    ? { width: args.viewportWidth, height: args.viewportHeight }
    : args.orientation === "portrait"
      ? { width: 1080, height: 1620 }
      : { width: 1920, height: 1080 };

  fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
  fs.mkdirSync(path.resolve(args.artifactDir), { recursive: true });

  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const launchOptions = {
    headless: !args.headed,
    args: args.softwareRendering
      ? ["--use-gl=angle", "--use-angle=swiftshader"]
      : ["--ignore-gpu-blocklist"],
  };
  if (!args.softwareRendering && fs.existsSync(chromePath)) {
    launchOptions.executablePath = chromePath;
  }
  const browser = await chromium.launch(launchOptions);
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1,
    acceptDownloads: true,
  });
  const recordingStartedAt = Date.now();
  const page = await context.newPage();
  if (args.cpuThrottle > 1) {
    const cdpSession = await context.newCDPSession(page);
    await cdpSession.send("Emulation.setCPUThrottlingRate", {
      rate: args.cpuThrottle,
    });
  }
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push({ type: "console.error", text: message.text() });
    }
  });
  page.on("pageerror", (error) => {
    consoleErrors.push({ type: "pageerror", text: String(error) });
  });
  await page.addInitScript(() => {
    document.addEventListener("DOMContentLoaded", () => {
      const style = document.createElement("style");
      style.textContent = "html,body,canvas,*{cursor:none!important}";
      document.documentElement.appendChild(style);
    }, { once: true });
  });

  await page.goto(args.url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(
    () => typeof window.render_game_to_text === "function",
    null,
    { timeout: 60000 },
  );
  await page.waitForFunction(
    () => {
      try {
        return JSON.parse(window.render_game_to_text()).mode === "lobby";
      } catch {
        return false;
      }
    },
    null,
    { timeout: 60000 },
  );
  await page.waitForTimeout(800);
  await page.screenshot({
    path: path.join(args.artifactDir, `${args.orientation}-lobby.png`),
  });

  const startPoint = args.orientation === "portrait"
    ? { x: viewport.width - 165, y: viewport.height - 153 }
    : { x: viewport.width * 0.875, y: viewport.height * 0.789 };
  await page.mouse.click(startPoint.x, startPoint.y);
  await page.waitForFunction(
    () => {
      try {
        return JSON.parse(window.render_game_to_text()).mode === "match";
      } catch {
        return false;
      }
    },
    null,
    { timeout: 10000 },
  );
  const gameplayStartedAt = Date.now();
  await startPerformanceSampling(page);
  const canvasCapture = await startCanvasRecording(page, 60, args.codec);
  await playPreviewRoute(page);
  const performanceSummary = await stopPerformanceSampling(page);
  const captureResult = await stopCanvasRecording(page, args.output);
  const graphicsInfo = await page.evaluate(() => {
    const canvas = [...document.querySelectorAll("canvas")]
      .sort((left, right) => (
        (right.width * right.height) - (left.width * left.height)
      ))[0];
    const gl = canvas?.getContext("webgl2") || canvas?.getContext("webgl");
    if (!gl) return {};
    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    return {
      renderer: gl.getParameter(gl.RENDERER),
      vendor: gl.getParameter(gl.VENDOR),
      unmasked_renderer: debugInfo
        ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
        : "",
      unmasked_vendor: debugInfo
        ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
        : "",
    };
  });
  const finalState = readGameState(
    await page.evaluate(() => window.render_game_to_text()),
  );
  await page.screenshot({
    path: path.join(args.artifactDir, `${args.orientation}-gameplay.png`),
  });

  const metadata = {
    orientation: args.orientation,
    viewport,
    recording_start_unix_ms: recordingStartedAt,
    gameplay_start_seconds: (
      (gameplayStartedAt - recordingStartedAt) / 1000
    ),
    gameplay_capture_seconds: (Date.now() - gameplayStartedAt) / 1000,
    launch_mode: args.softwareRendering ? "software" : "hardware",
    headed: args.headed,
    cpu_throttle: args.cpuThrottle,
    graphics: graphicsInfo,
    capture: {
      ...canvasCapture,
      ...captureResult,
    },
    performance_summary: performanceSummary,
    final_state: finalState,
    console_errors: consoleErrors,
  };
  fs.writeFileSync(
    path.join(args.artifactDir, `${args.orientation}-metadata.json`),
    `${JSON.stringify(metadata, null, 2)}\n`,
  );

  await context.close();
  await browser.close();

  if (consoleErrors.length > 0) {
    throw new Error(`Browser errors: ${JSON.stringify(consoleErrors)}`);
  }
  process.stdout.write(`${JSON.stringify(metadata, null, 2)}\n`);
}


main().catch((error) => {
  console.error(error);
  process.exit(1);
});
