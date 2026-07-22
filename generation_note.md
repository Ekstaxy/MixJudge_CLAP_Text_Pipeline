# 混音助手 (MixAssist) 訓練資料生成與架構規劃

## 一、 資料生成輸入與輸出結構 (Input & Output)

為了避免模型產生幻覺 (Hallucination)，必須將結構化的地標籤轉化為自然對話。

### 1. Input 結構 (Ground Truth)
輸入必須是高度結構化的 JSON，包含四個核心屬性：
* **Axis (問題維度):** 例如 Level, Body, Brightness, Masking 等。
* **Dim (具體症狀):** 例如 too_loud, muddy, harsh, swamping 等。
* **Subject (問題主體):** Figure (通常是 Vocal) 或 Bed (伴奏)。
* **Severity (嚴重程度):** Mild, Medium, Severe。
> **範例:** `{"axis": "Level", "dim": "too_loud", "subject": "figure", "severity": "medium"}`

### 2. Output 結構 (生成的訓練資料)
包含兩種 Persona，以控制對話中的「感知廢話」與「專業術語」比例：
* **Amateur (Client):** 負責提出問題（如："聽起來人聲好像有點太突兀了，感覺沒有跟音樂融合在一起？"）
* **Expert (Mixer):** 負責解答並點出核心與做法（如："沒錯，Vocal 的 Level 明顯太大聲了。我們把主唱的推桿稍微拉下來一點..."）

---

## 二、 Style Transfer 提示詞模板 (Prompt Template)

```text
# 任務指令
你是一位專業的音訊工程對話資料生成器。你的任務是將「結構化的混音問題標籤」，轉換為一段錄音室內 Amateur (客戶/新手) 與 Expert (混音師) 的自然對話 (Style Transfer)。

# 角色設定
- Amateur: 聽得出聲音有問題，但只會使用感知詞彙 (如：糊糊的、太刺、被吃掉、感覺不對)，不會使用具體的混音術語。說話帶有 30% 的閒聊或猶豫語氣。
- Expert: 語氣專業、友善且精準。必須在回覆中明確點出對應的「混音問題 (Problem Axis)」，並給出符合邏輯的處理建議 (Fix Action)。

# 轉換規則 (Style Guidelines)
1. 必須嚴格遵循 Input 給定的 Axis, Dim, Subject 與 Severity。
2. 對話的長度與廢話程度 (Verbosity) 設定為：適中 (Medium)。保留一點口語化的自然感，但不要過度冗長。
3. Amateur 提出問題，Expert 解答並點出核心。

# 範例 (Few-Shot Examples)
Input: {"axis": "Body", "dim": "muddy", "subject": "bed", "severity": "medium"}
Output:
Amateur: "這首歌的伴奏聽起來有點悶悶的，好像全部糊在一起，沒什麼精神耶？"
Expert: "我聽到了。伴奏 (Bed) 的中低頻累積太多，導致整體聽起來比較 Muddy (混濁)。我們可以在伴奏的群組軌切掉一點 250Hz 左右的頻段，聲音就會乾淨很多。"

# 現在，請轉換以下 Input：
Input: {插入結構化標籤}
Output: