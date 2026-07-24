# MixAssist 對話標註流程

為了讓後續的 L2 style transfer / retrieval 能夠建立在可驗證的混音問題標籤上，我們先以大型語言模型對 MixAssist 錄音室對話進行結構化標註。目標不是直接生成新對話，而是把每個 turn 中可支持的「感知問題」（problem）與「對應處置」（fix）抽成固定的 axis–dimension 表示，並保留完整分佈以供後續篩選。

---

## 1. 流程總覽

圖 1 呈現從 MixAssist 原始對話到 L2 problem pool 的完整管線。資料以 turn 為單位進入；系統為每個 turn 組裝含主題、有限歷史與當前回合的 prompt，交由量化後的 Gemma 指令模型推論；模型以 JSON 回傳一組或多組標籤後，管線再補上 axis、展開成表格列，並在後處理階段濾除無問題樣本、合併 train / validation / test，供下游生成使用。

<!-- 原圖極寬（約 11k px），預覽會縮成一條細線；此處改嵌直向拼接版以便閱讀。原圖見 Labeling_Flow_Chart.png -->
![圖 1：MixAssist 標註管線總覽](Labeling_Flow_Chart_stacked.png)

**圖 1.** MixAssist labeling pipeline（由上至下對應原圖由左至右）：來源資料 → prompt 組裝 → LLM 推論 → JSON 結構化輸出 → 正規化與一對多展開 → problem pool 匯出 → L2 生成。完整橫向原圖見 [`Labeling_Flow_Chart.png`](Labeling_Flow_Chart.png)。

整條管線可概括為三層意圖：

1. **標註**：盡可能忠實地把對話中的混音問題與處置寫成結構化欄位；即使該 turn 無問題，仍寫入結果，以保留真實分佈。
2. **正規化**：模型只負責感知維度（dimension）；對應的粗粒度軸（axis）由既定對照表決定，避免模型混淆兩者。
3. **匯出**：僅在進入 L2 前，才保留「確實存在問題」的樣本並合併各 split。

---

## 2. 輸入與上下文設計

MixAssist 每一筆樣本對應 Amateur（user）與 Expert（assistant）的一輪對話，並附有主題（topic）以及該回合之前的輸入歷史（input_history）。標註時我們不把單一句子孤立丟給模型，而是刻意構造可消歧義的上下文，原因在於混音對話高度依賴指代與延續——例如「把它調小聲」必須靠先前輪次才能判斷「它」指的是人聲還是軍鼓。

送給模型的 user message 由下列區塊組成：

**（1）TOPIC。** 當前 session 的粗粒度主題（如 drums）。主題提供背景，但不視為某一 stem 存在的充分證據。

**（2）INPUT HISTORY（最多最近 4 個 earlier turns）。** 歷史以 Amateur / Expert 成對排版，僅作輔助：當當前回合用代名詞、未說完的意圖、或延續先前同一混音決策時，用來判斷 stem、dimension，並在必要時回收 `problem_text`。歷史不是第二份獨立標註來源；若當前回合已切換到另一議題，不應把舊問題重新標進本 turn。

**（3）CURRENT TURN。** 當前 Amateur 與 Expert 的原文，作為 `problem_text` 與 `fix_text` 的主要證據。標註原則上必須能被當前回合的文字支持；模型被要求保守——工作流程閒聊、純音樂偏好或未陳述缺陷的「加效果建議」不應標成問題。

**（4）SYSTEM PROMPT。** 定義標註任務、證據規則、八個 axis 下的允許 dimensions、以及必須回傳的 JSON schema。其中特別強調：`problem_stem` 是「聽起來有問題／被掩蓋的樂器」，`fix_stem` 是「實際被操作的軌道」——兩者可以不同（例如軍鼓被鈸蓋住時，problem 在 snare，fix 卻可能作用在 cymbals）。

當某一方訊息為資料集中的占位句（Amateur 的 *Please analyze this audio segment.*，或 Expert 的 *I need more information...*）時，prompt 會附加說明：該方本回合未真正發言，應改以另一方與歷史判斷是否仍能形成有效標籤。

最終，system prompt 與上述 user message 一併餵入本地量化模型。

---

## 3. 結構化輸出表示

### 3.1 JSON 標籤 schema

模型被約束為只輸出 JSON，且頂層為 `labels` 陣列。每一個元素對應一個可區分的問題（或「無問題」的空標籤），主要欄位如下：

```json
{
  "labels": [
    {
      "has_problem": true,
      "problem_text": "",
      "problem_stem": "",
      "problem_dimension": "none",
      "problem_speaker": "",
      "vocal_lead": false,
      "has_fix": false,
      "fix_text": "",
      "fix_stem": "",
      "fix_action": "",
      "fix_speaker": "",
      "has_speaker": false,
      "label_reasoning": "",
      "confidence": "low"
    }
  ]
}
```

幾個設計選擇需要說明。第一，`has_problem` 與 `has_fix` 彼此獨立：可能只有處置、沒有新提出的問題，也可能只有問題陳述而無明確操作。第二，當既無問題也無 fix 時，回傳恰好一筆空標籤，且 `problem_dimension` 為 `none`。第三，若同一 turn 清楚出現兩個不同問題（即使共用同一個 fix），則 `labels` 含兩個元素，後續會展開為表格中的兩列。第四，模型**不輸出** `problem_axis`——只輸出較細的 `problem_dimension`，axis 由對照表決定，以降低把 dimension 名稱誤當 axis 的錯誤。

### 3.2 Axis 與 Dimension 詞彙

我們採用八個混音感知軸，每個軸對應一組有符號方向的 dimensions（phase 僅單向；另含 `none` 表示無支持問題）：

| Axis | Dimensions |
|------|------------|
| level | too_quiet, too_loud |
| body | muddy, thin |
| brightness | harsh, dull |
| space | too_wet, too_dry |
| dynamic | over_compressed, under_compressed |
| masking | swamping, invading |
| stereo | too_wide, too_narrow |
| phase | phase |

`problem_stem` / `fix_stem` 則限制在一組固定樂器與 `mix` 等詞彙上（如 vocal、kick、snare、cymbals、ambience 等）；無法對應者歸為 `other`。`vocal_lead` 另以布林標示該標籤是否涉及主唱（lead vocal），以區分 backing vocals 等同為 vocal stem 的情形。

`confidence` 分為 low / mid / high，用以反映證據強度：直接明文較高；需仰賴歷史才能還原受影響 stem 或延續問題者居中；仍有用但模糊者偏低。

---

## 4. 解析、正規化與表格化

模型原始字串經過清洗後才寫入資料集：去除可能夾帶的 Markdown code fence、修補常見 JSON 瑕疵，並在輸出被截斷時盡量救回已完整的 label 物件。接著依 dimension 反查 axis，並對 stem、布林與 confidence 做正規化。

由於一個 turn 可對應多個 labels，輸出表格採「一問題一列」：列之間共享相同的 `conversation_id` 與 `turn_id`，但各自攜帶不同的 `problem_dimension` / `problem_stem` 等。例如同一回合同時指出大鼓太小聲與鈸太吵，便會成為兩列。

需強調：此階段**不丟棄** `problem_dimension = none` 的列。完整標註結果本身即是對對話分佈的記錄；若在標註當下就過濾，將失去「無問題 turn」所占比例等資訊。

表格中除標籤欄位外，也保留對齊與抽查所需的中繼資料，例如 split、topic、音檔路徑，以及 Amateur / Expert 原文。核心標籤欄位包括：是否有問題／處置、problem 與 fix 各自的 stem、axis、dimension、speaker、原文片段、vocal_lead、fix_action、簡短推理與信心度。

---

## 5. 匯出 L2 Problem Pool

進入 L2 生成之前，我們另做一層匯出，只留下適合作為 exemplar 的「有問題」樣本。篩選條件要求：標註過程無錯誤、`has_problem` 為真、`problem_dimension` 存在且不為 `none`，且 `problem_text` 非空（可再選擇提高最低 confidence）。各 split 可先各自輸出 problem-only 檔，再串接為合併檔（如 `all_problems.csv`），作為後續 L2 style transfer 的 retrieval pool。

匯出欄位相對精簡，聚焦問題側（axis、dimension、stem、problem_text 等）與對齊音訊所需的識別欄；fix 相關欄位在此階段可省略，因為 L2 的核心是依問題狀態生成或檢索 Amateur／Expert 風格表述，而非重放處置細節。

---
