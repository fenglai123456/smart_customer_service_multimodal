const form = document.querySelector("#serviceForm");
const textInput = document.querySelector("#textInput");
const voiceInput = document.querySelector("#voiceInput");
const audioInput = document.querySelector("#audioInput");
const imageInput = document.querySelector("#imageInput");
const emotionImageInput = document.querySelector("#emotionImageInput");
const fileName = document.querySelector("#fileName");
const audioName = document.querySelector("#audioName");
const emotionImageName = document.querySelector("#emotionImageName");
const imagePreview = document.querySelector("#imagePreview");
const emotionImagePreview = document.querySelector("#emotionImagePreview");
const audioPreview = document.querySelector("#audioPreview");
const fillDemo = document.querySelector("#fillDemo");
const resetForm = document.querySelector("#resetForm");
const handoffStatus = document.querySelector("#handoffStatus");
const ticketMeta = document.querySelector("#ticketMeta");
const intentLabel = document.querySelector("#intentLabel");
const confidenceValue = document.querySelector("#confidenceValue");
const replyText = document.querySelector("#replyText");
const stepList = document.querySelector("#stepList");
const reasonList = document.querySelector("#reasonList");
const signalList = document.querySelector("#signalList");
const rankingBars = document.querySelector("#rankingBars");
const fusionFeatureList = document.querySelector("#fusionFeatureList");
const fusionStatus = document.querySelector("#fusionStatus");
const demoCaseList = document.querySelector("#demoCaseList");
const caseDetail = document.querySelector("#caseDetail");
const runCaseButton = document.querySelector("#runCaseButton");
const serviceCard = document.querySelector("#serviceCard");
const resultPanel = document.querySelector(".result-panel");

// Frontend state: demo cases come from /api/demo-cases, while object URLs are
// used only for local previews of files manually uploaded during the recording.
let demoCases = [];
let selectedCase = null;
const objectUrls = {
  image: "",
  emotionImage: "",
  audio: "",
};

const serviceMap = {
  consult: {
    title: "订单/服务咨询接待",
    detail: "优先展示订单状态、物流进度或服务说明，保持自动接待。",
  },
  complaint: {
    title: "投诉升级与人工跟进",
    detail: "记录负向情绪和投诉原因，必要时触发人工客服介入。",
  },
  refund: {
    title: "退款售后流程",
    detail: "核对订单状态，进入退款申请、售后审核和到账进度查询。",
  },
  repair: {
    title: "故障报修工单",
    detail: "收集截图和异常描述，创建故障工单并给出排查步骤。",
  },
  howto: {
    title: "操作引导服务",
    detail: "输出分步操作说明，引导用户完成设置或入口查找。",
  },
};

loadDemoCases();

// Quick-fill button for manual demonstrations; the one-click demo case flow
// below remains the recommended path for the final recorded video.
fillDemo.addEventListener("click", () => {
  textInput.value = "页面上传截图后一直报错，我已经付款但订单显示失败，想申请退款并找人工处理。";
  voiceInput.value = "我现在有点着急，麻烦尽快帮我看一下。";
  document.querySelector('input[name="emotion"][value="anxious"]').checked = true;
});

resetForm.addEventListener("click", () => {
  form.reset();
  setFileLabel(fileName, "上传客服截图/故障页面");
  setFileLabel(audioName, "支持 WAV、MP3、FLAC");
  setFileLabel(emotionImageName, "上传 FER-2013 人脸样例");
  clearAllPreviews();
  renderEmpty();
});

imageInput.addEventListener("change", () => {
  renderImagePreview(imageInput, fileName, imagePreview, "image", "上传客服截图/故障页面", "未选择业务截图");
});

audioInput.addEventListener("change", () => {
  renderAudioPreview(audioInput, audioName, audioPreview, "audio", "支持 WAV、MP3、FLAC", "未选择语音文件");
});

emotionImageInput.addEventListener("change", () => {
  renderImagePreview(
    emotionImageInput,
    emotionImageName,
    emotionImagePreview,
    "emotionImage",
    "上传 FER-2013 人脸样例",
    "未选择表情图片",
  );
});

runCaseButton.addEventListener("click", async () => {
  if (!selectedCase) return;
  setLoading(true, "正在运行演示用例...");
  try {
    const response = await fetch(`/api/demo-cases/${selectedCase.case_id}/predict`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`演示用例运行失败：${response.status}`);
    }
    renderResult(await response.json());
  } catch (error) {
    replyText.textContent = error.message || "演示用例运行失败，请检查服务。";
  } finally {
    setLoading(false);
  }
});

// Manual upload path: send text, ASR text, audio, expression image and business
// screenshot to /api/predict so the backend can run the same multimodal fusion.
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true, "正在融合文字、语音转写、音频声学、表情和图片特征...");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: new FormData(form),
    });

    if (!response.ok) {
      throw new Error(`请求失败：${response.status}`);
    }

    renderResult(await response.json());
  } catch (error) {
    replyText.textContent = error.message || "识别失败，请检查服务是否已启动。";
  } finally {
    setLoading(false);
  }
});

async function loadDemoCases() {
  try {
    const response = await fetch("/api/demo-cases");
    if (!response.ok) {
      throw new Error("演示数据读取失败");
    }
    const data = await response.json();
    demoCases = data.cases || [];
    renderDemoCases();
    if (demoCases.length) {
      selectCase(demoCases[0]);
    }
  } catch (error) {
    demoCaseList.innerHTML = `<button class="demo-case-card" type="button" disabled>${escapeHtml(error.message)}</button>`;
  }
}

function renderDemoCases() {
  demoCaseList.innerHTML = "";
  demoCases.forEach((demoCase) => {
    const button = document.createElement("button");
    button.className = "demo-case-card";
    button.type = "button";
    button.dataset.caseId = demoCase.case_id;
    button.innerHTML = `
      <strong>${escapeHtml(demoCase.scenario)}</strong>
      <small>预期：${escapeHtml(demoCase.expected_intent_label)} · ${escapeHtml(demoCase.folder)}</small>
    `;
    button.addEventListener("click", () => selectCase(demoCase));
    demoCaseList.appendChild(button);
  });
}

function selectCase(demoCase) {
  // Selecting a course demo case fills all visible controls and shows previews
  // of the screenshot, expression image and audio file before prediction.
  selectedCase = demoCase;
  document.querySelectorAll(".demo-case-card").forEach((button) => {
    button.classList.toggle("active", button.dataset.caseId === demoCase.case_id);
  });
  textInput.value = demoCase.text || "";
  voiceInput.value = demoCase.voice_text || "";
  const emotionInput = document.querySelector(`input[name="emotion"][value="${demoCase.emotion}"]`);
  if (emotionInput) emotionInput.checked = true;
  imageInput.value = "";
  emotionImageInput.value = "";
  audioInput.value = "";
  setFileLabel(fileName, demoCase.screenshot_file || "上传客服截图/故障页面");
  setFileLabel(emotionImageName, demoCase.emotion_image_file || "上传 FER-2013 人脸样例");
  setFileLabel(audioName, demoCase.audio_file || "支持 WAV、MP3、FLAC");
  renderDemoFilePreviews(demoCase);
  runCaseButton.disabled = false;
  caseDetail.innerHTML = `
    <span>演示文件</span>
    <p>${escapeHtml(demoCase.note || "完整多模态测试用例。")}</p>
    <code title="截图：${escapeHtml(demoCase.screenshot_file)}">截图：${escapeHtml(demoCase.screenshot_file)}</code>
    <code title="表情：${escapeHtml(demoCase.emotion_image_file)}">表情：${escapeHtml(demoCase.emotion_image_file)}</code>
    <code title="语音：${escapeHtml(demoCase.audio_file)}">语音：${escapeHtml(demoCase.audio_file)}</code>
  `;
}

function renderImagePreview(input, labelEl, previewEl, key, placeholder, emptyText) {
  const file = input.files[0];
  clearPreview(key);
  if (!file) {
    setFileLabel(labelEl, placeholder);
    renderEmptyPreview(previewEl, emptyText);
    return;
  }

  const url = URL.createObjectURL(file);
  objectUrls[key] = url;
  setFileLabel(labelEl, file.name);
  previewEl.classList.remove("is-empty");
  previewEl.innerHTML = "";
  const image = document.createElement("img");
  image.src = url;
  image.alt = `${file.name} 预览`;
  previewEl.appendChild(image);
}

function renderAudioPreview(input, labelEl, previewEl, key, placeholder, emptyText) {
  const file = input.files[0];
  clearPreview(key);
  if (!file) {
    setFileLabel(labelEl, placeholder);
    renderEmptyPreview(previewEl, emptyText);
    return;
  }

  const url = URL.createObjectURL(file);
  objectUrls[key] = url;
  setFileLabel(labelEl, file.name);
  previewEl.classList.remove("is-empty");
  previewEl.innerHTML = "";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = url;
  audio.preload = "metadata";
  previewEl.appendChild(audio);
}

function renderDemoFilePreviews(demoCase) {
  // Packaged demo files are served by Flask, so previews work without requiring
  // the user to manually select local files during class presentation.
  clearAllPreviews();
  renderRemoteImagePreview(imagePreview, demoCase.file_urls?.screenshot, demoCase.screenshot_file, "演示业务截图");
  renderRemoteImagePreview(emotionImagePreview, demoCase.file_urls?.emotion_image, demoCase.emotion_image_file, "演示表情图片");
  renderRemoteAudioPreview(audioPreview, demoCase.file_urls?.audio, demoCase.audio_file);
}

function renderRemoteImagePreview(previewEl, url, filename, altPrefix) {
  if (!url || !filename) {
    renderEmptyPreview(previewEl, "演示图片缺失");
    return;
  }
  previewEl.classList.remove("is-empty");
  previewEl.innerHTML = "";
  const image = document.createElement("img");
  image.src = url;
  image.alt = `${altPrefix}：${filename}`;
  previewEl.appendChild(image);
}

function renderRemoteAudioPreview(previewEl, url, filename) {
  if (!url || !filename) {
    renderEmptyPreview(previewEl, "演示音频缺失");
    return;
  }
  previewEl.classList.remove("is-empty");
  previewEl.innerHTML = "";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = url;
  audio.preload = "metadata";
  previewEl.appendChild(audio);
}

function renderEmptyPreview(previewEl, message) {
  previewEl.classList.add("is-empty");
  previewEl.innerHTML = "";
  const text = document.createElement("span");
  text.textContent = message;
  previewEl.appendChild(text);
}

function clearPreview(key) {
  if (objectUrls[key]) {
    URL.revokeObjectURL(objectUrls[key]);
    objectUrls[key] = "";
  }
}

function clearAllPreviews() {
  Object.keys(objectUrls).forEach(clearPreview);
  renderEmptyPreview(imagePreview, "未选择业务截图");
  renderEmptyPreview(emotionImagePreview, "未选择表情图片");
  renderEmptyPreview(audioPreview, "未选择语音文件");
}

function setFileLabel(element, value) {
  element.textContent = value;
  element.title = value;
}

function renderResult(data) {
  // The backend response contains both business-facing fields and technical
  // explanation fields; this renderer keeps them visible for the report video.
  const prediction = data.prediction;
  const percent = Math.round(prediction.confidence * 100);

  ticketMeta.textContent = `${data.ticket_id} · ${data.created_at}`;
  intentLabel.textContent = prediction.intent_label;
  confidenceValue.textContent = `${percent}%`;
  replyText.textContent = prediction.response.reply;

  handoffStatus.textContent = prediction.need_human ? "转人工" : "自动接待";
  handoffStatus.classList.toggle("warn", prediction.need_human);

  renderService(prediction);
  renderSteps(prediction.response.steps);
  renderReasons(prediction.reasons);
  renderSignals(prediction.modality_signals);
  renderRanking(prediction.ranking);
  renderFusionDetails(prediction.modality_signals?.fusion);
  scrollResultIntoView();
}

function renderService(prediction) {
  const service = serviceMap[prediction.intent] || {
    title: prediction.intent_label,
    detail: "根据当前融合结果继续接待。",
  };
  serviceCard.innerHTML = `
    <span>推荐服务</span>
    <strong>${escapeHtml(service.title)}</strong>
    <p>${escapeHtml(service.detail)}</p>
  `;
}

function renderSteps(steps) {
  stepList.innerHTML = "";
  steps.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    stepList.appendChild(li);
  });
}

function renderReasons(reasons) {
  reasonList.innerHTML = "";
  reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonList.appendChild(li);
  });
}

function renderSignals(signals) {
  // Summarize each modality so the page proves text, audio, expression image,
  // screenshot and fusion signals all participated in the final decision.
  const image = signals.image;
  const audio = signals.audio;
  const asr = signals.asr;
  const emotionImage = signals.emotion_image;
  const textModel = signals.text_model;
  const fusion = signals.fusion;
  const screenshot = image?.details?.screenshot_demo_decision || image?.details?.screenshot_intent_cnn;
  const screenshotText = screenshot ? `；截图五分类：${escapeHtml(screenshot.class_name)} ${Math.round((screenshot.confidence || 0) * 100)}%` : "";

  signalList.innerHTML = `
    <dt>文字</dt>
    <dd>${signals.text.provided ? `${signals.text.chars} 个字符` : "未输入"}${textModel?.label ? `；文本模型：${escapeHtml(textModel.label)}` : ""}</dd>
    <dt>语音</dt>
    <dd>${signals.voice.provided ? escapeHtml(signals.voice.transcript) : "未输入"}${audio ? `；声学 ${Math.round(audio.confidence * 100)}%` : ""}${asr?.text ? `；ASR：${escapeHtml(asr.text)}` : ""}</dd>
    <dt>表情</dt>
    <dd>${emotionImage ? `${escapeHtml(emotionImage.label)} ${Math.round(emotionImage.confidence * 100)}%` : signals.emotion.summary}</dd>
    <dt>截图</dt>
    <dd>${image ? `${escapeHtml(image.label)} ${Math.round(image.confidence * 100)}%${screenshotText}` : "未上传"}</dd>
    <dt>融合</dt>
    <dd>${fusion ? `${escapeHtml(fusion.source)}：${escapeHtml(fusion.summary)}` : "未启用"}</dd>
  `;
}

function renderRanking(ranking) {
  rankingBars.innerHTML = "";
  ranking.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${escapeHtml(item.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width: ${Math.round(item.score * 100)}%"></span></span>
      <strong>${Math.round(item.score * 100)}%</strong>
    `;
    rankingBars.appendChild(row);
  });
}

function renderFusionDetails(fusion) {
  // Display the required "feature concatenation + fully connected fusion" route:
  // text5 + audio5 + emotion5 + image5 + numeric6 = 26-dimensional MLP input.
  if (!fusion || !fusion.details) {
    fusionStatus.textContent = "未启用";
    fusionStatus.classList.remove("warn");
    fusionFeatureList.innerHTML = "<p>提交后展示 26 维拼接特征和 MLP 融合结构。</p>";
    return;
  }

  const details = fusion.details;
  const blocks = Array.isArray(details.feature_blocks) ? details.feature_blocks : [];
  const layerText = Array.isArray(details.mlp_layers) && details.mlp_layers.length
    ? details.mlp_layers.join(" → ")
    : "26 → 48 → 24 → 5";
  const statusText = fusion.source === "fully_connected_fusion" ? "MLP 已启用" : "兜底融合";

  fusionStatus.textContent = `${statusText} · ${details.feature_dim || 0}维`;
  fusionStatus.classList.toggle("warn", fusion.source !== "fully_connected_fusion");
  fusionFeatureList.innerHTML = `
    <div class="fusion-summary-grid">
      <div class="fusion-stat">
        <span>拼接公式</span>
        <strong>文字5 + 语音5 + 表情5 + 截图5 + 数值6 = ${escapeHtml(details.feature_dim || 26)}维</strong>
      </div>
      <div class="fusion-stat">
        <span>全连接层</span>
        <strong>${escapeHtml(layerText)}</strong>
      </div>
    </div>
    <div class="fusion-flow" aria-label="特征拼接流">
      <span>文字 5</span>
      <span>语音 5</span>
      <span>表情 5</span>
      <span>截图 5</span>
      <span>数值 6</span>
      <strong>MLP</strong>
    </div>
    <div class="fusion-block-grid">
      ${blocks.map((block) => renderFeatureBlock(block, details.label_names || {})).join("")}
    </div>
  `;
}

function renderFeatureBlock(block, labelNames) {
  const rangeText = Array.isArray(block.range) ? `#${block.range[0]}-${block.range[1]}` : `${block.dim || 0}维`;
  if (block.key === "numeric") {
    const featureOrder = Array.isArray(block.feature_order) ? block.feature_order : Object.keys(block.values || {});
    const entries = featureOrder.map((name) => [name, block.values?.[name] || 0]);
    return `
      <article class="feature-block numeric-feature-block">
        <div class="feature-block-header">
          <strong>${escapeHtml(block.name || "辅助数值特征")}</strong>
          <small>${escapeHtml(rangeText)}</small>
        </div>
        <details class="feature-details">
          <summary>查看 6 个辅助特征</summary>
          <div class="numeric-chip-row">
            ${entries.map(([name, value]) => `
              <span class="numeric-chip">
                ${escapeHtml(shortFeatureName(name))} ${formatScore(value)}
              </span>
            `).join("")}
          </div>
        </details>
      </article>
    `;
  }

  const values = block.values || {};
  const labels = Array.isArray(block.label_order) ? block.label_order : Object.keys(values);
  const top = block.top || {};
  return `
      <article class="feature-block">
        <div class="feature-block-header">
          <strong>${escapeHtml(shortBlockName(block.key, block.name))}</strong>
          <small>${escapeHtml(rangeText)}</small>
        </div>
      <div class="feature-top">
        <span>最高</span>
        <b>${escapeHtml(top.label_name || labelNames[top.label] || top.label || "-")}</b>
        <em>${formatPercent(top.score)}</em>
      </div>
      <details class="feature-details">
        <summary>查看 5 类分数</summary>
        <div class="mini-score-tags">
          ${labels.map((label) => {
            const value = Number(values[label] || 0);
            return `
              <span>${escapeHtml(labelNames[label] || label)} ${formatPercent(value)}</span>
            `;
          }).join("")}
        </div>
      </details>
    </article>
  `;
}

function renderEmpty() {
  ticketMeta.textContent = "等待用户输入";
  intentLabel.textContent = "未识别";
  confidenceValue.textContent = "0%";
  replyText.textContent = "提交后将根据多模态特征生成回复与引导流程。";
  handoffStatus.textContent = "自动接待";
  handoffStatus.classList.remove("warn");
  serviceCard.innerHTML = "<span>推荐服务</span><strong>等待识别</strong><p>提交后展示系统推荐的客服动作。</p>";
  stepList.innerHTML = "<li>等待识别结果</li>";
  reasonList.innerHTML = "<li>暂无分析</li>";
  signalList.innerHTML = "<dt>文字</dt><dd>未输入</dd><dt>语音</dt><dd>未上传</dd><dt>表情图</dt><dd>未上传</dd><dt>截图</dt><dd>未上传</dd><dt>融合</dt><dd>未启用</dd>";
  rankingBars.innerHTML = "";
  renderFusionDetails(null);
}

function setLoading(isLoading, message = "") {
  form.classList.toggle("loading", isLoading);
  runCaseButton.disabled = isLoading || !selectedCase;
  if (message) {
    replyText.textContent = message;
  }
}

function scrollResultIntoView() {
  if (!window.matchMedia("(max-width: 1360px)").matches) return;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatScore(value) {
  return Number(value || 0).toFixed(2);
}

function shortFeatureName(name) {
  return {
    text_length_ratio: "文本长度",
    has_text: "文字",
    has_voice_text: "转写",
    audio_confidence: "语音置信",
    emotion_image_confidence: "表情置信",
    image_confidence: "截图置信",
  }[name] || name;
}

function shortBlockName(key, name) {
  return {
    text: "文字分支",
    audio: "声学分支",
    emotion: "情绪分支",
    image: "截图分支",
  }[key] || name || "模态分支";
}
