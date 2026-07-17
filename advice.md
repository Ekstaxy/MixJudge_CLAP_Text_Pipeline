用白話講：這整段計畫其實只有三件事——**先搞懂資料、再做成一張有標籤的表、最後證明這張表比 Jason 好**。下面按「你要幹嘛 / 產出是什麼 / 為什麼」拆開。

---

## Phase 1：先讀書、動手標一點（還沒寫管線）

**目的：** 你自己先搞清楚「什麼叫 mud、什麼叫 space_wet」，以及 MixAssist 對話長什麼樣。還沒跑 LLM 全量抽取。

| 步驟 | 你實際做什麼 | 產出 |
|---|---|---|
| 讀 9 keys | 打開 `intern_chores.md` 那張表，把每個 key 的「聽起來像什麼」記下來 | 腦中／筆記裡的定義 |
| 讀 Jason CSV | 打開 `problem_fix_gold_pairs.csv`（和 meaningful 那份），挑 10 筆覺得好的、10 筆覺得爛的，寫為什麼 | 錯誤類型清單（例如：problem/fix 對不上、dimension 亂標） |
| 看 `train.csv` | 看 `has_content`、各 `topic`、user/assistant 誰在抱怨誰在給建議 | 對資料長什麼樣有感覺 |
| 寫 mapping | 一張對照表：口語 → key。例：`too much reverb / washed out` → `space_wet` | **對齊規則**（之後給 LLM 當說明） |
| 人工標 30–50 turns | 自己選一些 turn，標：有沒有 problem/fix、是哪個 key、原文哪一句 | **seed gold**（小答案卡，用來之後檢查 LLM 對不對） |

**這步不是**整份 labeled CSV。  
**這步是**「考試前先自己做幾題標準答案」。

---

## Phase 2a：用 LLM 幫你填那張「很寬的 CSV」（你本來想做的）

**目的：** 對每個有內容的 turn，抽出 problem / fix，並標 9 keys。這就是你說的：retrieve 前先有很多 column。

對每個 `has_content=True` 的 turn，LLM 填類似這些欄：

- 原始：`conversation_id`, `turn_id`, `user`, `assistant`, `audio_file`, `topic`
- 抽取：`problem_text`, `fix_text`, `problem_speaker`, `fix_speaker`
- 標籤：`retrieval_key`（9 個之一或 `none`）
- 品質：`has_problem`, `has_fix`, `confidence`, `reasoning`
- 規則：problem 和 fix 必須來自**同一個 turn**；若 fix 是「A 或 B」就拆成兩列

人工抽樣：每個 key 看 5–10 筆，prompt 寫爛就改，直到錯誤率可接受。

**產出兩個檔（概念上）：**

1. **annotations** — 所有跑過的 turn（含 `none`、只有 problem 沒 fix）
2. **pairs** — 只要「有 problem + 有 fix + key 不是 none」的列（之後 L3、跟 Jason 比用）

**用你的話說：** 這步做完，你就有那張寬表了；之後「retrieval」= 在這張表上用條件篩選／排序。

---

## Phase 2b：我之前寫的 embedding / cross-encoder 是什麼？

這步容易糊，直接講清楚：

**2b 不是你的主路徑必做項。**  
它是另一種做法：不先抽句子，而是用向量模型對 raw turn 搜「跟 mud 最像的對話」。

| 做法 | 怎麼找「屬於 mud 的 turns」 |
|---|---|
| **你的做法（2a + 條件篩選）** | 先標好 `retrieval_key=mud`，再 `WHERE key==mud AND has_fix` |
| **2b embedding** | 用「mud 的定義」當 query，去 raw 文本裡找相似 turn，再排序 |

對你現在的計畫：**可以先不做 2b**。  
Phase 2 deliverable「每個 key 一個 ranked pool」用你的方式也能交：

```text
從 pairs/annotations 裡
  篩 retrieval_key == 某個 key
  再依 confidence（或之後加的 relevance score）排序
  存成該 key 的 pool
```

若之後發現某個 key 標太少、或想補漏，再加 embedding 當「第二路召回」即可。

---

## Phase 2c：怎麼證明你比 Jason 好（具體在比什麼）

不是嘴說「我比較準」，而是固定比法：

1. **對齊標籤**：把他 CSV 裡的 `frequency_mud`、`level_too_loud` 等，對照成你的 9 keys；對不上的算 `other`，不拿來搶功。
2. **每個 key 各拿 top-k**（例如各 10 筆）：你的 pool vs 他 mapped 後該 key 的樣本。
3. **盲審**：給人（或另一個 LLM）看「這段 raw 對話 + 這個 key」，只答「相關 / 不相關」，不要先說是誰產的。
4. **算數字**：Precision@k = 相關筆數 / k；順便看 9 個 key 你有沒有都蓋到。
5. **寫例子**：5 個他對你錯、5 個你對他錯，報告才站得住。

**統計圖**就是：每個 key 有幾筆、有幾筆成對、誰說的 problem/fix——用來找 sparse key（例如 masking 幾乎沒有）。

---

## 收成一條你能照做的線（對齊你的原意）

```text
Phase 1  搞懂 + 手標 30~50 題當標準答案
    ↓
Phase 2a  LLM 填寬表（problem/fix/key/speaker…）← 你的主工作
    ↓
「Retrieval」= 依 key / has_fix / confidence 篩選排序 → 每 key 一個 pool
    ↓
Phase 2c  跟 Jason 盲審比 Precision@k
```

**2b（embedding）先當成可選加強，不是必做。**

---

## 一句話對照「模糊詞」

| 我之前寫的詞 | 實際意思 |
|---|---|
| seed gold | 你手標的 30–50 筆小答案集 |
| schema | 寬 CSV 有哪些欄位 |
| annotations | 全量標註結果（含不合格／none） |
| pairs | 同 turn 且 problem+fix 都有的列 |
| retrieval pool | 某個 key 篩出來、排好序的那一疊 turn |
| Precision@k | top-k 裡有幾筆真的跟該 key 相關 |
| mapping | 口語／Jason 舊標籤 → 你的 9 keys |

若你要，下一步我可以只寫一版「寬 CSV 欄位清單（每一欄填什麼、範例一列）」——仍然用白話、對應你 extract-then-filter 的流程。