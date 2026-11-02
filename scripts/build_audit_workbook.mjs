import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const root = process.cwd();
const auditDir = path.join(root, "results/audit/v1_v2_csv_lineage_2026-08-24");
const outputDir = path.join(root, "outputs/lncrna-audit-2026-08-24");
const renderDir = path.join(outputDir, "renders");

await fs.mkdir(renderDir, { recursive: true });

const summaryCsv = [
  "V1/V2 CSV and feature-provenance audit,,,,,,,,",
  "Evidence snapshot generated 2026-08-24,,,,,,,,",
].join("\n");
const workbook = await Workbook.fromCSV(summaryCsv, { sheetName: "Summary" });

const imports = [
  ["CSV Lineage", "csv_lineage.csv"],
  ["Feature Inventory", "feature_inventory.csv"],
  ["Findings", "audit_findings.csv"],
  ["Paper Block", "paper_binary_block_evidence.csv"],
  ["RF Probe", "rf_probe_comparison.csv"],
  ["Rerun Checks", "rerun_reproducibility.csv"],
];
for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(path.join(auditDir, fileName), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

const summary = workbook.worksheets.getItem("Summary");
summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A2:H2").merge();
summary.getRange("A1").values = [["V1/V2 CSV and feature-provenance audit"]];
summary.getRange("A2").values = [["What is wrong, what is proven, and which clean RF configuration should be used"]];

summary.getRange("A4:E4").values = [["Final clean matrices", "Samples", "Features", "Labels", "Positive associations"]];
summary.getRange("A5:A6").values = [["V1 strict"], ["V2 transcripts"]];
summary.getRange("B5:E6").formulas = [
  ["='CSV Lineage'!F8", "='CSV Lineage'!K8", "='CSV Lineage'!J9", "='CSV Lineage'!L9"],
  ["='CSV Lineage'!F22", "='CSV Lineage'!K22", "='CSV Lineage'!J23", "='CSV Lineage'!L23"],
];

summary.getRange("A9:E9").values = [["Matched 44-label artifact audit", "Feature dimensions", "AUROC", "AUPRC", "F1 @ 0.5"]];
summary.getRange("A10:E12").values = [
  ["Paper artifact: continuous remainder", 542, 0.633, 0.082, 0.009],
  ["Clean V1: all 2,023 features", 2023, 0.661, 0.097, 0.032],
  ["Clean V1: core sequence", 583, 0.667, 0.104, 0.030],
];

summary.getRange("A15:E15").values = [["Paired few-fold RF probe", "Features", "AUROC", "AUPRC", "Training-OOB F1"]];
summary.getRange("A16:A17").formulas = [["='RF Probe'!A2"], ["='RF Probe'!A3"]];
summary.getRange("B16:E17").formulas = [
  ["='RF Probe'!B2", "='RF Probe'!D2", "='RF Probe'!E2", "='RF Probe'!G2"],
  ["='RF Probe'!B3", "='RF Probe'!D3", "='RF Probe'!E3", "='RF Probe'!G3"],
];

summary.getRange("A20:H20").merge();
summary.getRange("A20").values = [["Audit conclusion"]];
summary.getRange("A21:H22").merge();
summary.getRange("A21").values = [[
  "The public 827-column V1 artifact must be quarantined because it contains a 285-column target-like binary block. " +
  "This establishes a defect in the supplied artifact, not proof that the published scores used that exact file."
]];
summary.getRange("A24:H25").merge();
summary.getRange("A24").values = [[
  "The very low clean F1 is primarily a threshold-calibration problem. The implemented correction learns one global cutoff " +
  "from each outer training fold's Random Forest out-of-bag predictions; test labels are never used to choose it."
]];
summary.getRange("A27:H28").merge();
summary.getRange("A27").values = [[
  "Do not restore PseDNC/DAC/DCC/DACC from the current dinuc_props.csv: it contains corpus count/frequency values rather than " +
  "a validated physicochemical property table. Historical V2 gene-region features are superseded by transcript cDNA features."
]];

const dark = "#0F172A";
const teal = "#0F766E";
const paleTeal = "#CCFBF1";
const paleBlue = "#E0F2FE";
const paleAmber = "#FEF3C7";
const white = "#FFFFFF";
const grid = "#CBD5E1";

summary.getRange("A1:H1").format = {
  fill: dark,
  font: { bold: true, color: white, size: 18 },
  verticalAlignment: "center",
};
summary.getRange("A2:H2").format = {
  fill: "#E2E8F0",
  font: { italic: true, color: "#334155" },
  verticalAlignment: "center",
};
for (const range of ["A4:E4", "A9:E9", "A15:E15", "A20:H20"]) {
  summary.getRange(range).format = {
    fill: teal,
    font: { bold: true, color: white },
    borders: { preset: "all", style: "thin", color: grid },
  };
}
summary.getRange("A5:E6").format = { borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("A10:E12").format = { borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("A16:E17").format = { borders: { preset: "all", style: "thin", color: grid } };
summary.getRange("C10:E12").format.numberFormat = "0.000";
summary.getRange("C16:E17").format.numberFormat = "0.000";
summary.getRange("B5:E6").format.numberFormat = "#,##0";
summary.getRange("A21:H28").format = {
  wrapText: true,
  verticalAlignment: "center",
  fill: paleAmber,
  borders: { preset: "outside", style: "thin", color: "#D97706" },
};
summary.getRange("A10:E10").format.fill = paleAmber;
summary.getRange("A11:E12").format.fill = paleTeal;
summary.getRange("A16:E17").format.fill = paleBlue;
summary.getRange("A1:H28").format.rowHeight = 22;
summary.getRange("A1:H1").format.rowHeight = 34;
summary.getRange("A21:H22").format.rowHeight = 44;
summary.getRange("A24:H25").format.rowHeight = 44;
summary.getRange("A27:H28").format.rowHeight = 44;
summary.getRange("A:A").format.columnWidth = 42;
summary.getRange("B:E").format.columnWidth = 18;
summary.getRange("F:H").format.columnWidth = 14;
summary.freezePanes.freezeRows(2);

const sheetSpecs = [
  ["CSV Lineage", "A1:M24", "LineageTable"],
  ["Feature Inventory", "A1:L68", "FeatureInventoryTable"],
  ["Findings", "A1:E7", "FindingsTable"],
  ["Paper Block", "A1:B14", "PaperBlockTable"],
  ["RF Probe", "A1:J3", "RFProbeTable"],
  ["Rerun Checks", "A1:H21", "RerunChecksTable"],
];
for (const [sheetName, tableRange, tableName] of sheetSpecs) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getRange(tableRange);
  used.format.font = { name: "Aptos", size: 10 };
  used.format.verticalAlignment = "top";
  used.format.borders = { preset: "all", style: "thin", color: "#E2E8F0" };
  sheet.getRange(tableRange.split(":")[0].replace(/\d+$/, "1") + ":" + tableRange.split(":")[1].replace(/\d+$/, "1")).format = {
    fill: dark,
    font: { bold: true, color: white },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.tables.add(tableRange, true, tableName);
}

const lineage = workbook.worksheets.getItem("CSV Lineage");
lineage.getRange("A:A").format.columnWidth = 9;
lineage.getRange("B:B").format.columnWidth = 26;
lineage.getRange("C:C").format.columnWidth = 48;
lineage.getRange("D:D").format.columnWidth = 62;
lineage.getRange("E:L").format.columnWidth = 15;
lineage.getRange("M:M").format.columnWidth = 34;
lineage.getRange("C2:D24").format.wrapText = true;

const features = workbook.worksheets.getItem("Feature Inventory");
features.getRange("A:C").format.columnWidth = 20;
features.getRange("D:D").format.columnWidth = 44;
features.getRange("E:J").format.columnWidth = 16;
features.getRange("K:K").format.columnWidth = 62;
features.getRange("L:L").format.columnWidth = 62;
features.getRange("D2:D68").format.wrapText = true;
features.getRange("K2:L68").format.wrapText = true;
features.getRange("I2:I68").conditionalFormats.add("containsText", {
  text: "eligible", format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
});
features.getRange("I2:I68").conditionalFormats.add("containsText", {
  text: "excluded", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
});
features.getRange("I2:I68").conditionalFormats.add("containsText", {
  text: "quarantined", format: { fill: paleAmber, font: { color: "#92400E", bold: true } },
});

const findings = workbook.worksheets.getItem("Findings");
findings.getRange("A:A").format.columnWidth = 10;
findings.getRange("B:B").format.columnWidth = 46;
findings.getRange("C:E").format.columnWidth = 68;
findings.getRange("B2:E7").format.wrapText = true;
findings.getRange("A2:E2").format.fill = "#FEE2E2";

const paper = workbook.worksheets.getItem("Paper Block");
paper.getRange("A:A").format.columnWidth = 58;
paper.getRange("B:B").format.columnWidth = 20;
paper.getRange("B2:B14").format.numberFormat = "0.000000";

const probe = workbook.worksheets.getItem("RF Probe");
probe.getRange("A:A").format.columnWidth = 30;
probe.getRange("B:I").format.columnWidth = 18;
probe.getRange("J:J").format.columnWidth = 62;
probe.getRange("D2:I3").format.numberFormat = "0.000";

const checks = workbook.worksheets.getItem("Rerun Checks");
checks.getRange("A:F").format.columnWidth = 20;
checks.getRange("G:H").format.columnWidth = 34;
checks.getRange("E2:E21").conditionalFormats.add("containsText", {
  text: "identical", format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
});
checks.getRange("E2:E21").conditionalFormats.add("containsText", {
  text: "pending", format: { fill: paleAmber, font: { color: "#92400E", bold: true } },
});
checks.getRange("E2:E21").conditionalFormats.add("containsText", {
  text: "mismatch", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
});

const keyInspect = await workbook.inspect({
  kind: "region",
  sheetId: "Summary",
  range: "A1:H28",
  maxChars: 6000,
});
console.log(keyInspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of ["Summary", ...imports.map(([name]) => name)]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(renderDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "lncrna_v1_v2_audit_2026-08-24.xlsx");
await output.save(outputPath);
console.log(outputPath);
