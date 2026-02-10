const form = document.getElementById("inferForm");
const promptFileInput = document.getElementById("promptFile");
const promptTextArea = document.getElementById("promptText");
const responseText = document.getElementById("responseText");
const statusBadge = document.getElementById("statusBadge");
const runButton = document.getElementById("runButton");

function setBadge(state, text) {
  statusBadge.className = `badge ${state}`;
  statusBadge.textContent = text;
}

function bindRangeValue(inputId, outputId, formatter = (value) => value) {
  const input = document.getElementById(inputId);
  const output = document.getElementById(outputId);
  const update = () => {
    output.textContent = formatter(input.value);
  };
  input.addEventListener("input", update);
  update();
  return update;
}

const updateFrameLimit = bindRangeValue("frameLimit", "frameLimitOut");
const updateMaxSide = bindRangeValue("maxSide", "maxSideOut");
const updateJpegQuality = bindRangeValue("jpegQuality", "jpegQualityOut");
const updateTemperature = bindRangeValue(
  "temperature",
  "temperatureOut",
  (value) => Number(value).toFixed(2)
);
const updateMaxOutputTokens = bindRangeValue("maxOutputTokens", "maxOutputTokensOut");
const updateTimeout = bindRangeValue("timeout", "timeoutOut");

promptFileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  try {
    const text = await file.text();
    promptTextArea.value = text.trim();
  } catch (error) {
    responseText.textContent = `Error: Could not read prompt file. ${error}`;
    setBadge("error", "Prompt Error");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  responseText.textContent = "";
  setBadge("running", "Running");
  runButton.disabled = true;
  runButton.textContent = "Processing...";

  const formData = new FormData(form);

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const errorMessage = payload.error || `Request failed with status ${response.status}.`;
      responseText.textContent = `Error: ${errorMessage}`;
      setBadge("error", "Error");
      return;
    }

    responseText.textContent = payload.response || "";
    setBadge("done", "Done");
  } catch (error) {
    responseText.textContent = `Error: ${error}`;
    setBadge("error", "Network Error");
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run Counting";
  }
});

form.addEventListener("reset", () => {
  requestAnimationFrame(() => {
    responseText.textContent = "";
    setBadge("idle", "Idle");
    updateFrameLimit();
    updateMaxSide();
    updateJpegQuality();
    updateTemperature();
    updateMaxOutputTokens();
    updateTimeout();
  });
});
