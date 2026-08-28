INSIGHT 1 — 年齡是非線性風險主因，60 歲後進入加速區間

Type: age_effect
Statement:

Mortality risk increases non-linearly with attained age, with a pronounced acceleration after approximately age 60. Individuals above age 83 (top 5%) represent an extreme-risk tail relative to the overall population.

Evidence:

Median age ≈ 53

75th percentile = 65

95th percentile = 83

Long right tail up to age 120

Implication for explanation:

對高齡個案，模型給出高風險屬於「結構性風險」，不是異常

83 歲應搭配「高齡尾端不確定性」提示

INSIGHT 2 — Duration 呈現明顯長尾，長期保單是風險與不確定性的交會區

Type: duration_effect
Statement:

Policy duration exhibits a strong right-skewed distribution. Policies beyond duration 32 years (top 5%) form a long-tail segment where selection effects have largely dissipated and model uncertainty increases.

Evidence:

Median duration = 12

90th percentile = 27

95th percentile = 32

Max = 119

Implication for explanation:

長 duration ≠ 單純「老保單」，而是「選擇效應消失後的真實風險區」

Duration >30 應提示「模型校準與資料支撐度需特別注意」

INSIGHT 3 — Issue Age 與 Attained Age 必須聯合解釋，單獨看會誤導

Type: age_consistency
Statement:

Issue age alone is insufficient to characterize risk. Its interpretation depends critically on duration and attained age, and inconsistent combinations may indicate data or cohort structure effects rather than true individual risk.

Evidence:

Issue_Age median = 40

Attained_Age ≈ Issue_Age + Duration（結構關係）

Implication for explanation:

LLM 解釋時應避免「因為投保年齡高所以風險高」的單因果說法

高 Issue_Age + 短 Duration ≠ 高 Attained_Age

INSIGHT 4 — Missing 並非隨機：Preferred_Class 的 NA 是一個主要族群

Type: missing_as_signal
Statement:

Preferred_Class has a dominant NA category (~32%), indicating that missingness reflects structural or programmatic differences rather than random data absence.

Evidence:

NA = 32.12%（單一最大類別）

多個等級同時存在 1 / 1.0、2 / 2.0（來源或編碼差異）

Implication for explanation:

NA 不應被描述為「一般風險」

涉及 Preferred_Class 的解釋應說明「未納入偏好體系或不適用」

INSIGHT 5 — Smoker_Status 的 Unknown 具 proxy 性質，不可忽略

Type: proxy_risk
Statement:

Approximately 8–9% of records have unknown smoking status, suggesting a proxy for data source, product type, or underwriting regime rather than random missingness.

Evidence:

Smoker = 19.1%

Non-smoker = 72.3%

Unknown = 8.6%

Implication for explanation:

Unknown ≠ Non-smoker

若 SHAP 對 Unknown 有顯著貢獻，應提示其「間接風險指標」性質

INSIGHT 6 — 商品型態高度不均，整體結論主要由 Term 驅動

Type: product_mix_bias
Statement:

Nearly half of the dataset consists of Term products, implying that population-level conclusions primarily reflect Term behavior and may not generalize to Permanent or Variable products.

Evidence:

Term ≈ 46%

Largest non-Term ≈ 16%

Implication for explanation:

解釋非 Term 個案時，需優先引用「同產品 segment」而非整體平均

避免用「整體趨勢」套用到小眾產品

INSIGHT 7 — 高保額族群極度稀有，解釋必須伴隨支撐度聲明

Type: rare_segment
Statement:

High face amount policies (≥5M) constitute a very small fraction of the data (<2.5%), indicating that model explanations for such cases rely on limited empirical support.

Evidence:

5M–10M ≈ 1.49%

≥10M ≈ 0.73%

Implication for explanation:

高保額個案需附加「低樣本支撐」或「不確定性較高」說明

避免過度解讀單筆 SHAP

INSIGHT 8 — Level Term 結構資訊高度不完整，NA 本身即為狀態

Type: structural_indicator
Statement:

Over half of the records are marked as N/A for level term indicators, implying that these variables primarily encode product structure applicability rather than missing data.

Evidence:

SOA_Post_Lvl_Ind NA ≈ 54%

Anticipated / Guaranteed TP N/A ≈ 54%

Implication for explanation:

解釋 WLT / PLT / ULT 時，需明示「僅適用於 level term 產品」

NA 應被視為「非 level term 狀態」

INSIGHT 9 — 極端類別（如 40-year term）不具可泛化性

Type: low_credibility
Statement:

Certain categorical levels (e.g., 40-year term periods) are extremely rare and should not be interpreted as stable risk patterns.

Evidence:

40-year term count < 100

Implication for explanation:

觸發低可信度 disclaimer

若模型對其給出極端預測，應標示為「資料支撐不足」

INSIGHT 10 — 整體資料呈現「結構風險 + 長尾不確定性」的典型壽險樣態

Type: global_structure
Statement:

The dataset reflects a typical life insurance risk structure: strong age and duration effects dominate central predictions, while long-tail segments (very old ages, long durations, high face amounts) contribute disproportionally to uncertainty.

Implication for explanation:

大多數預測屬於「穩定結構性風險」

少數極端輸入應優先解釋「不確定性」而非「確定風險」