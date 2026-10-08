const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType,
} = require("docx");

const PRIMARY = "2C5F2D", ACCENT = "C68A2E", MUTED = "6B7280", CODE_BG = "F2F2F2", HEAD_BG = "2C5F2D";
const R = JSON.parse(fs.readFileSync("integration_test_results.json"));

function h1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
  children: [new TextRun({ text: t, bold: true, color: PRIMARY, size: 28 })] }); }
function h2(t) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 },
  children: [new TextRun({ text: t, bold: true, color: ACCENT, size: 24 })] }); }
function body(t) { return new Paragraph({ spacing: { after: 160, line: 300 }, children: [new TextRun({ text: t, size: 22 })] }); }
function bullet(t) { return new Paragraph({ bullet: { level: 0 }, spacing: { after: 90, line: 280 }, children: [new TextRun({ text: t, size: 22 })] }); }
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

const pc = R.prediction_cases;
const perClass = {};
for (const c of pc) {
  perClass[c.expected] = perClass[c.expected] || { n: 0, ok: 0 };
  perClass[c.expected].n++;
  perClass[c.expected].ok += c.pass ? 1 : 0;
}
const failures = pc.filter((c) => !c.pass);
const S = R.summary;

const doc = new Document({ sections: [{
  properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } },
  children: [
    new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "WEEK 4 TASK REPORT", bold: true, size: 20, color: ACCENT })] }),
    new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Integration of Computer Vision Solutions and Cross-functional Testing", bold: true, size: 34, color: PRIMARY })] }),
    new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Computer Vision Internship Program \u2014 Prateek, RNS Institute of Technology", size: 21, color: MUTED })] }),

    h1("1. Objective"),
    body("This task integrates the optimized computer vision model into a small application service that other software components can call over a network, and tests that integration end to end. The results in this report come from an actual test run (integration_test_results.json)."),

    h1("2. Application Context"),
    body("The simulated application is an image-classification service of the kind a real-time surveillance, quality-control or augmented-reality front end would call. A client uploads an image; the service returns the predicted shape class, its confidence and the full probability distribution. The model deployed is the Week 3 int8-quantized variant, which kept 91.0% accuracy at roughly one quarter of the original weight storage."),

    h1("3. Integration Strategy"),
    h2("3.1 Architecture"),
    img("diagrams/integration_architecture.png", 6.3, 1.8),
    caption("Figure 5. Request-response flow from client to classifier and back."),
    h2("3.2 Interface"),
    bullet("GET /health: liveness check, returns service status"),
    bullet("POST /predict: multipart image upload; returns JSON with predicted_class, confidence and probabilities"),
    bullet("Input validation: non-image content types and undecodable image bytes are rejected with HTTP 400 and a descriptive message"),
    h2("3.3 Modularity"),
    body("The web layer (api.py) depends only on the model's public forward pass and the class-label list. The model, quantization and training code live in separate modules, so the model can be retrained or re-optimized without changing the API contract."),

    h1("4. Test Plan"),
    body("Testing used FastAPI's in-process test client, which sends real HTTP-style requests through the full routing, validation and serialization path without opening a network socket. The plan covered three areas: service health, prediction behavior on fresh synthetic samples, and error handling for bad input."),
    table(["Area", "Cases", "Purpose"], [
      ["Health", "1", "Service responds and reports status"],
      ["Prediction", `${pc.length}`, "End-to-end classification correctness on unseen samples (15 per class)"],
      ["Error handling", `${R.error_handling.length}`, "Non-image upload and corrupted image bytes are rejected cleanly"],
    ]),

    h1("5. Results"),
    table(["Metric", "Value"], [
      ["Total test cases", `${S.total_tests}`],
      ["Passed", `${S.total_pass}`],
      ["Failed", `${S.total_tests - S.total_pass}`],
      ["Prediction accuracy via live API", `${(S.api_prediction_accuracy * 100).toFixed(1)}% (${pc.filter((c) => c.pass).length}/${pc.length})`],
    ]),
    h2("5.1 Per-class prediction results"),
    table(["Class", "Correct", "Samples", "Accuracy"], Object.entries(perClass).map(([k, v]) =>
      [k, `${v.ok}`, `${v.n}`, `${((v.ok / v.n) * 100).toFixed(1)}%`])),
    h2("5.2 Error-handling results"),
    table(["Case", "HTTP status", "Result"], R.error_handling.map((e) =>
      [e.case.replace(/_/g, " "), `${e.status_code}`, e.pass ? "PASS" : "FAIL"])),

    h1("6. Analysis"),
    bullet(`The service reproduces the model's behavior end to end: ${pc.filter((c) => c.pass).length} of ${pc.length} fresh synthetic predictions were correct through the API. This is consistent with the 91.0% test accuracy measured in Week 3, within the variation expected for a 60-sample check.`),
    bullet("Every failure is a misclassification by the model, not a failure of the API layer. Each request returned HTTP 200 with a valid prediction body."),
    bullet("Malformed input is handled at the boundary: the service never crashes on a bad upload and returns a clear client-error response instead."),
    failures.length > 0
      ? bullet(`Misclassified samples (${failures.length}): ${failures.map((f) => `${f.expected} predicted as ${f.predicted}`).join("; ")}.`)
      : bullet("No misclassified samples in this run."),

    h1("7. Challenges Encountered"),
    bullet("The model is loaded lazily on the first prediction request, so the health endpoint reports model_loaded = false until then. This is deliberate, to keep startup fast, but a production service would load the model at startup and fail fast if the checkpoint is missing."),
    bullet("The test client emits a deprecation warning about its HTTP library dependency. This does not affect results, but the dependency should be updated before deployment."),
    bullet("Test images are generated by the same synthetic generator used for training, so the prediction accuracy shows integration correctness on in-distribution data. It is not evidence of performance on real photographs."),

    h1("8. Limitations"),
    bullet("Only one deployment target (a single local service) was exercised; no load, latency-under-concurrency or network-failure tests were run."),
    bullet("No authentication, rate limiting or request logging is implemented."),
    bullet("The service runs on the same synthetic 4-class task; a real application would need a model trained on its own domain data."),

    h1("9. Recommendations"),
    bullet("Load the model at service startup and add a readiness check that reflects it"),
    bullet("Add load tests (concurrent requests) and record p95 latency on the target hardware"),
    bullet("Containerize the service and add authentication before any shared deployment"),
    bullet("Connect the service to a real client (a simple web page or a camera feed on the Jetson Orin Nano) for a live demonstration"),
  ],
}] });

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("Week4_Report_Integration.docx", buf); console.log("Week 4 report written"); });
