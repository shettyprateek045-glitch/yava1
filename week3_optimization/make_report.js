const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType,
} = require("docx");

const PRIMARY = "2C5F2D", ACCENT = "C68A2E", MUTED = "6B7280", CODE_BG = "F2F2F2", HEAD_BG = "2C5F2D";

const R = JSON.parse(fs.readFileSync("benchmark_results.json"));
const f2 = (x) => x.toFixed(2), f3 = (x) => x.toFixed(3), f1 = (x) => x.toFixed(1);

function h1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
  children: [new TextRun({ text: t, bold: true, color: PRIMARY, size: 28 })] }); }
function h2(t) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 },
  children: [new TextRun({ text: t, bold: true, color: ACCENT, size: 24 })] }); }
function body(t) { return new Paragraph({ spacing: { after: 160, line: 300 }, children: [new TextRun({ text: t, size: 22 })] }); }
function bullet(t) { return new Paragraph({ bullet: { level: 0 }, spacing: { after: 90, line: 280 }, children: [new TextRun({ text: t, size: 22 })] }); }
function codeBlock(lines) {
  const b = { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" };
  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE },
    borders: { top: b, bottom: b, left: b, right: b },
    rows: [new TableRow({ children: [new TableCell({ shading: { type: ShadingType.CLEAR, fill: CODE_BG },
      margins: { top: 150, bottom: 150, left: 150, right: 150 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 },
        children: [new TextRun({ text: l || " ", font: "Consolas", size: 17, color: "1A1D29" })] })) })] })] });
}
function img(path, w, h) { return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w * 96, height: h * 96 } })] }); }
function caption(t) { return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
  children: [new TextRun({ text: t, italics: true, size: 19, color: MUTED })] }); }
function table(headers, rows) {
  const cell = (t, hdr) => new TableCell({
    shading: hdr ? { type: ShadingType.CLEAR, fill: HEAD_BG } : undefined,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: hdr, color: hdr ? "FFFFFF" : "1A1D29", size: 20 })] })] });
  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: [
    new TableRow({ children: headers.map((h) => cell(h, true)) }),
    ...rows.map((r) => new TableRow({ children: r.map((c) => cell(c, false)) })) ] });
}
const sp = R.pruned_40pct.layer_sparsity;
const logText = fs.readFileSync("optimization_log.txt", "utf-8");

const doc = new Document({ sections: [{
  properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } },
  children: [
    new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "WEEK 3 TASK REPORT", bold: true, size: 20, color: ACCENT })] }),
    new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Real-Time Performance Optimization of Computer Vision Algorithms", bold: true, size: 34, color: PRIMARY })] }),
    new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Computer Vision Internship Program \u2014 Prateek, RNS Institute of Technology", size: 21, color: MUTED })] }),

    h1("1. Objective"),
    body("This task takes the CNN trained in Week 2 and applies optimization techniques to make it faster and lighter, while measuring the accuracy cost. All figures in this report are taken from an actual benchmark run (benchmark_results.json), not estimated."),

    h1("2. Techniques Applied"),
    h2("2.1 Magnitude-based weight pruning"),
    body("Weights with the smallest absolute values are set to zero, per layer, targeting 40% sparsity. The intent is that small-magnitude weights contribute least to the output and can be removed with limited accuracy loss."),
    h2("2.2 Post-training int8 quantization"),
    body("Each layer's weights are mapped from float32 to 8-bit integers using a per-layer symmetric scale (max absolute weight / 127), then dequantized for the forward pass. This is a simulated (fake-quant) evaluation: it reproduces the accuracy effect of int8 storage, and the storage size reported below is computed for int8 weights with fp32 biases."),

    h1("3. Benchmark Setup"),
    bullet("Model: Week 2 CNN (67,252 parameters), loaded from model_checkpoint.npz"),
    bullet("Test set: 200 synthetic images (50 per class), resized to 32x32"),
    bullet("Latency: mean and standard deviation over 30 timed forward passes, batch size 32, after one warm-up pass"),
    bullet("Hardware/runtime: NumPy CPU implementation in the development sandbox; absolute times will differ on other machines, so compare the variants relative to each other"),

    h1("4. Results"),
    table(["Variant", "Test Accuracy", "Weight Size", "Inference (ms, batch 32)"], [
      ["Baseline (fp32)", f3(R.baseline.accuracy), `${f1(R.baseline.weight_size_kb)} KB`, `${f2(R.baseline.inference_ms_mean)} \u00b1 ${f2(R.baseline.inference_ms_std)}`],
      ["Pruned (40%)", f3(R.pruned_40pct.accuracy), `${f1(R.pruned_40pct.weight_size_kb)} KB (dense storage)`, `${f2(R.pruned_40pct.inference_ms_mean)} \u00b1 ${f2(R.pruned_40pct.inference_ms_std)}`],
      ["Quantized (int8)", f3(R.quantized_int8.accuracy), `${f1(R.quantized_int8.weight_size_kb)} KB`, `${f2(R.quantized_int8.inference_ms_mean)} \u00b1 ${f2(R.quantized_int8.inference_ms_std)}`],
    ]),
    img("benchmark_comparison.png", 6.3, 2.2),
    caption("Figure 4. Accuracy, weight storage and inference time for each variant (actual benchmark run)."),

    h2("4.1 Pruning sparsity achieved per layer"),
    table(["Layer", "Weights zeroed"], Object.entries(sp).map(([k, v]) => [k, `${(v * 100).toFixed(1)}%`])),

    h1("5. Analysis and Trade-offs"),
    bullet(`Quantization is the clear winner on size: weight storage fell by ${f1(R.quantized_int8.size_reduction_pct)}% (${f1(R.baseline.weight_size_kb)} KB to ${f1(R.quantized_int8.weight_size_kb)} KB) for a ${Math.abs((R.quantized_int8.accuracy - R.baseline.accuracy) * 100).toFixed(1)}-point accuracy drop.`),
    bullet(`Pruning cost more accuracy (${((R.pruned_40pct.accuracy - R.baseline.accuracy) * 100).toFixed(1)} points) and gave no storage saving in this implementation, because the zeroed weights are still stored densely. Realizing a size or speed gain from pruning requires a sparse storage format or sparse kernels.`),
    bullet("Inference time did not improve for either variant in this NumPy build. This is expected: pruning leaves the dense matrix multiply unchanged, and quantization's fake-quant evaluation dequantizes back to float32 before computing, so neither removes arithmetic. The observed differences (12.5 to 13.6 ms) are within run-to-run noise (standard deviations of roughly 1 to 1.6 ms), so no speed conclusion should be drawn from them."),
    bullet("A real latency gain from int8 quantization requires an integer inference runtime (for example TensorFlow Lite, ONNX Runtime with int8 kernels, or a hardware NPU). This report therefore claims the storage reduction and the accuracy trade-off as measured, and treats speed as a deployment-time property to be measured on the target hardware."),

    h1("6. Honest Limitations"),
    bullet("Accuracy figures are on a 200-image synthetic test set; differences of 1 to 4 points correspond to only 2 to 8 images and are not statistically strong."),
    bullet("Timing was measured on a single CPU sandbox with NumPy; it is a relative comparison only."),
    bullet("Quantization is simulated; it measures accuracy impact, not an integer-kernel speedup."),

    h1("7. Recommendations for Future Work"),
    bullet("Deploy quantized weights through an integer runtime (TensorFlow Lite or ONNX Runtime) and benchmark FPS on the target device, such as the Jetson Orin Nano"),
    bullet("Use structured (channel-level) pruning instead of unstructured pruning, so the removed weights produce real speed and size gains on standard hardware"),
    bullet("Fine-tune after pruning to recover accuracy lost in the 40% sparsity step"),
    bullet("Repeat measurements over multiple test sets and random seeds to report confidence intervals"),
  ],
}] });

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("Week3_Report_Optimization.docx", buf); console.log("Week 3 report written"); });
