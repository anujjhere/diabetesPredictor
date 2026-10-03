// Where the API from Step 3 is running
const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("risk-form");
const resultBox = document.getElementById("result");
const button = form.querySelector("button");
const ageSelect = form.querySelector('select[name="age"]');

// 0.618 -> "about 62%"; tiny values -> "under 1%" (avoids fake precision)
function formatRisk(risk) {
  const pct = risk * 100;
  return pct < 1 ? "under 1%" : "about " + Math.round(pct) + "%";
}

function showResult(cssClass, lines) {
  resultBox.className = cssClass;
  resultBox.replaceChildren(); // clear old content
  lines.forEach(({ text, cls }) => {
    const p = document.createElement("p");
    if (cls) p.className = cls;
    p.textContent = text; // textContent is safe: it never runs HTML
    resultBox.appendChild(p);
  });
  resultBox.hidden = false;
}

// Under 18: the model was trained on adults only, so give guidance, not a score
function checkAge() {
  if (ageSelect.value === "under18") {
    showResult("info", [
      { text: "This screener is built for adults (18 and over)." },
      {
        text:
          "It was trained on survey data from adults, so it cannot give a " +
          "reliable estimate for children or teenagers, whose risk factors " +
          "and common type of diabetes can differ. If you or a parent are " +
          "worried about symptoms such as unusual thirst, frequent urination " +
          "or unexplained weight loss, please speak to a doctor.",
      },
    ]);
    button.disabled = true;
    button.textContent = "Not available under 18";
    return true;
  }
  resultBox.hidden = true;
  button.disabled = false;
  button.textContent = "Check my risk";
  return false;
}

ageSelect.addEventListener("change", checkAge);
checkAge(); // also run on page load, in case the browser restored an old selection

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stop the page from reloading
  if (checkAge()) return;

  const data = new FormData(form);

  // The model needs BMI, so we compute it from height and weight
  const heightM = Number(data.get("height")) / 100;
  const weightKg = Number(data.get("weight"));
  const bmi = Math.round((weightKg / (heightM * heightM)) * 10) / 10;

  if (bmi < 10 || bmi > 100) {
    showResult("error", [
      { text: "Those height and weight values look unusual. Please check them." },
    ]);
    return;
  }

  // Only these 8 fields go to the model. Family history is NOT one of them.
  const person = {
    Age: Number(data.get("age")),
    BMI: bmi,
    HighBP: Number(data.get("highbp")),
    HighChol: Number(data.get("highchol")),
    PhysActivity: Number(data.get("physactivity")),
    GenHlth: Number(data.get("genhlth")),
    Sex: Number(data.get("sex")),
    Smoker: Number(data.get("smoker")),
  };

  button.disabled = true;
  button.textContent = "Checking...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(person),
    });
    if (!response.ok) {
      throw new Error("The server rejected those answers. Please check them.");
    }
    const result = await response.json();

    const familyYes = data.get("family") === "yes";
    const label = familyYes
      ? "Model estimate (does not include family history): "
      : "Estimated risk: ";
    const lines = [{ text: label + formatRisk(result.risk), cls: "risk" }];

    // "Keep up healthy habits" would sound like an all-clear next to a family history
    if (!(familyYes && !result.high_risk)) {
      lines.push({ text: result.message });
    }
    if (familyYes) {
      lines.push({
        text:
          "You mentioned a close relative with diabetes. Family history " +
          "raises risk, but this estimate does not include it, so your real " +
          "risk may be higher than shown. Consider a blood test even if " +
          "the result looks low.",
      });
    }
    lines.push({ text: result.note });

    // A low score plus family history shouldn't look like an all-clear
    const cssClass = result.high_risk ? "high" : familyYes ? "info" : "low";
    showResult(cssClass, lines);
  } catch (error) {
    const unreachable = error instanceof TypeError; // fetch failed to connect
    showResult("error", [
      {
        text: unreachable
          ? "Could not reach the server. Is the API still running?"
          : error.message,
      },
    ]);
  } finally {
    button.disabled = false;
    button.textContent = "Check my risk";
  }
});