try:
    import pysrt
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pysrt"])
import os
import pysrt  # ライブラリが未インストールの場合は "pip install pysrt" が必要
import openai
import csv
from io import StringIO
import shutil

# ▼ 環境変数 OPENAI_API_KEY に APIキーを設定してください
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_structured_output(text_block):
    import re
    fields = ["headline", "overview", "category", "tags", "stance", "timestamp"]
    # --- fast‑path: if the block looks like a CSV record (no 【】) -------------
    if "【" not in text_block and "," in text_block:
        import csv, io
        # take the first non-empty line
        first = next((ln for ln in text_block.splitlines() if ln.strip()), "")
        if first:
            row = next(csv.reader([first]))
            # pad or trim to exactly 6 columns
            if len(row) < 6:
                row += ["NULL"] * (6 - len(row))
            return row[:6]
    result = {field: "NULL" for field in fields}
    for line in text_block.splitlines():
        match = re.match(r"^【(.*?)】(.*)", line.strip())
        if match:
            key_raw, value = match.groups()
            key = key_raw.strip().lower()
            if key in result:
                result[key] = value.strip()
    return [result[field] for field in fields]

# ---------------------------------------------------------------------
#  Stance normalization utilities
# ---------------------------------------------------------------------
STANCE_CANON = {
    # --- 導入決定 ---
    "導入決定": "導入決定",
    "契約":     "導入決定",
    "予算計上": "導入決定",
    # --- 導入済み ---
    "導入済み": "導入済み",
    "運用開始": "導入済み",
    "本格運用": "導入済み",
    # --- 内部決定・制度化 ---
    "条例":        "内部決定・制度化",
    "制定":        "内部決定・制度化",
    "施行":        "内部決定・制度化",
    "規則":        "内部決定・制度化",
    "要綱":        "内部決定・制度化",
    "改正条例":    "内部決定・制度化",
    "内部決定":    "内部決定・制度化",
    "事業廃止":    "内部決定・制度化",
    "打ち切り":    "内部決定・制度化",
    "廃止":        "内部決定・制度化",
    "廃止決定":    "内部決定・制度化",
    "廃止予定":    "内部決定・制度化",
    "終了":        "内部決定・制度化",
    "終了予定":    "内部決定・制度化",
    # --- 前向き ---
    "前向き":  "前向き",
    "積極的":  "前向き",
    "推進":    "前向き",
    # --- 検討中 ---
    "前向きに検討": "検討中",
    "積極的に検討": "検討中",
    "検討中": "検討中",
    "協議":   "検討中",
    "議論":   "検討中",
    # --- 調査・情報収集段階 ---
    "動向を見て":   "調査・情報収集段階",
    "動向を注視":   "調査・情報収集段階",
    "国の政策":     "調査・情報収集段階",
    "情報収集":     "調査・情報収集段階",
    "事例調査":     "調査・情報収集段階",
    # --- 慎重・消極的 ---
    "慎重":       "慎重・消極的",
    "様子見":     "慎重・消極的",
    "財源の動向": "慎重・消極的",
    "導入困難":   "慎重・消極的",
    # --- 反対・否定 ---
    "反対": "反対・否定",
    "否定": "反対・否定",
    "不要": "反対・否定",
    "否決": "反対・否定",
    # --- 情報不足・判断不能 ---
    "情報不足": "情報不足・判断不能",
    "不明":     "情報不足・判断不能",
    "言及なし": "情報不足・判断不能",
    # --- 実証・試行段階 ---
    "実証":     "実証・試行段階",
    "試行":     "実証・試行段階",
    "モデル事業": "実証・試行段階",
    # --- 要望・提案段階 ---
    "要望": "要望・提案段階",
    "提案": "要望・提案段階",
    "陳情": "要望・提案段階",
    "請願": "要望・提案段階",
}

def normalize_stance(text: str) -> str:
    """
    与えられた文字列を STANCE_CANON の正規ラベルに変換。
    マッチしない場合は元の文字列を返す。
    """
    if not text or text.upper() == "NULL":
        return "情報不足・判断不能"
    # 長いフレーズほど優先的にマッチさせる
    for key in sorted(STANCE_CANON.keys(), key=len, reverse=True):
        if key in text:
            return STANCE_CANON[key]
    return text.strip()

def filter_meaningful_content(text_chunk):
    """
    "このテキストに少しでも発言や文章が含まれていたら 'Yes' と答えてください。"
    "ほぼ空っぽで何もないなら 'No' と答えてください。\n\n"
    """
    short_prompt = (
        "このテキストに少しでも議論や問題提起、発言が含まれているなら 'Yes'、"
        "全く何も無いなら 'No' と答えてください。\n\n"
        + text_chunk
    )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role": "user", "content": short_prompt}],
        max_tokens=5,
        temperature=0.0
    )
    result_text = completion.choices[0].message.content.strip()
    return result_text
############################################
# SRTの字幕を (index, text) のリストにする関数
############################################
def parse_srt_as_subs(file_path: str):
    """
    pysrtを使ってSRTファイルを読み込み、
    (字幕インデックス, 字幕テキスト) のタプルをまとめたリストを返す。
    sub.index: 1,2,3... (SRT上の連番)
    sub.text : 字幕本文
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        srt_content = f.read()
    subs = pysrt.from_string(srt_content)
    result = []
    for sub in subs:
        start_time_str = f"{sub.start.hours:02}:{sub.start.minutes:02}:{sub.start.seconds:02},{sub.start.milliseconds:03}"
        result.append((sub.index, start_time_str, sub.text))
    return result

############################################
# 字幕を 60分単位でチャンク化しつつ、
# 先頭のインデックスを "timestamp" として保持
############################################
def chunk_srt_subs_with_timestamp(subs_list, max_minutes=60):
    """
    Break subtitles into chunks whose total duration does not exceed *max_minutes*.
    Each chunk keeps the timestamp of its first subtitle as `timestamp`.

    Parameters
    ----------
    subs_list : list[tuple]
        List of tuples (index, start_time_str, sub_text) where *start_time_str*
        is `"HH:MM:SS,mmm"`.
    max_minutes : int, optional
        Maximum duration of one chunk, in minutes.  Default is 60 (≈1 hour).

    Returns
    -------
    list[dict]
        `[{'text': str, 'timestamp': str}, ...]`
        *text*  – concatenated subtitle blocks in this chunk  
        *timestamp* – start time of the first subtitle in the chunk
    """
    max_duration_sec = max_minutes * 60

    chunks = []
    current_text = []
    first_timestamp_in_chunk = None
    first_time_sec = None

    for (_, start_time_str, sub_text) in subs_list:
        # Parse start_time_str "HH:MM:SS,mmm" → seconds (float)
        h, m, rest = start_time_str.split(':')
        s, ms = rest.split(',')
        current_sec = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        # Initialise the first timestamp of a new chunk
        if first_time_sec is None:
            first_time_sec = current_sec
            first_timestamp_in_chunk = start_time_str

        # If adding this subtitle would exceed the chunk duration limit,
        # close the current chunk and start a new one.
        if current_sec - first_time_sec >= max_duration_sec and current_text:
            chunks.append({
                'text': ''.join(current_text),
                'timestamp': first_timestamp_in_chunk
            })
            current_text = []
            first_time_sec = current_sec
            first_timestamp_in_chunk = start_time_str

        # Append current subtitle block
        current_text.append(f"{start_time_str} {sub_text}\n")

    # Flush the final chunk
    if current_text:
        chunks.append({
            'text': ''.join(current_text),
            'timestamp': first_timestamp_in_chunk
        })

    return chunks

############################################
# summarize_chunk: "timestamp" を追加
############################################
def summarize_chunk(text_chunk, system_prompt, timestamp):
    """
    - text_chunk: このチャンク内の字幕テキスト
    - system_prompt: ユーザー独自の長い議事録要約プロンプト (下記参照)
    - timestamp: チャンク内で最初に登場した字幕のインデックス
    """
    prompt_input = f"""
    あなたは議事録の要約に特化したAIアシスタントです。以下の指示を厳守し、与えられた字幕テキスト（議事録）を分類・要約してください。

━━━━━━━━━━━━━━
【目的】
自治体関係者および民間事業者が、ヘッドラインと要約だけで「自分たちに関係のある議題かどうか」を瞬時に判別できるようにする。

━━━━━━━━━━━━━━
【出力フォーマット（CSV形式）】
ヘッダーを含む 6 列で 1 行出力してください。カンマ区切り、改行コードは "\n"。

headline,overview,category,tags,stance,timestamp

## 各列の要件

◆ headline（60文字以内）

* テーマの概要が一目でわかる短いタイトル。

◆ overview（700文字以上）

* 背景（問題意識・経緯）
* 「誰が何を提案・指摘・質問し、誰がどう答えたか」
* 今後の方向性（導入済みか・検討中か・否定されたか等）
* 議論の要点や結論が語られた時間を本文中に必ず "(HH\:MM\:SS)" 形式で挿入
* 冗長な挨拶・定型句は除く

◆ category（1 つ）
次の 20 分類から最も適切な 1 つを記入。
子育て・保育 / 学校教育・生涯学習 / 福祉・包摂（高齢・障がい・困窮） / 医療・公衆衛生 / 防災・危機管理・安全安心 / 環境・エネルギー / 経済・雇用・産業振興 / 農林水産業 / 都市整備・土地利用（ハード） / インフラ・公共施設 / 交通・モビリティ / デジタル・ICT推進 / 行政手続・窓口サービス / 財政・税務 / 総務・人事・組織運営 / 政策立案・企画・計画 / 議会・選挙・ガバナンス / 地域活性・コミュニティ（ソフト） / 市民協働・広報 / 人権・男女共同参画（ダイバーシティ）

◆ tags（最大 3 つ）

* 議論が活発だった主要キーワードをカンマ区切りで最大 3 つ。
* 該当しない場合は NULL。

◆ stance（1 つ）

* 次の 6 つのいずれかを記入（日本語）。

  1. 導入済み・決定済み
  2. 前向き・推進意向
  3. 検討中・調査中
  4. 慎重・消極的
  5. 否定・反対
  6. 判断困難・情報不足

◆ timestamp

* 議論の結論や方向性が示された時間帯を "HH\:MM\:SS〜HH\:MM\:SS" 形式で記入。

━━━━━━━━━━━━━━
【スタンス判定に関する注意事項】

* 数値（利用者数・予算額など）が登場しても、それ自体は導入状況を示す根拠にはなりません。発言者の意図・態度に着目してください。
* 具体的に「導入済み」「制度化が決定」「今後導入する方針」などが示されているかどうかを優先判断します。

━━━━━━━━━━━━━━
【入力テンプレート】
▼ 以下の <<TEXT>> を置き換えて実行

＜入力例＞

```
【TEXT】
<<ここに字幕テキスト（原文チャンク）をそのまま貼り付ける>>
```

━━━━━━━━━━━━━━
【出力例】
headline,overview,category,tags,stance,timestamp
"子ども食堂への支援拡大案","(00:12:30) 地域の子ども食堂への財政支援の必要性について議論。山田議員が孤食問題と経済格差の影響を挙げて支援拡大を提案。市側は現状の支援策を説明しつつ、他自治体の事例も参考に柔軟に対応していきたいと回答。(00:14:50) 市長は『予算調整が必要だが、前向きに検討したい』と発言。複数議員から利用者数の把握と事後評価の必要性が指摘され、今後、令和7年度予算編成の中で具体化を目指す。","福祉・包摂（高齢・障がい・困窮）","子ども食堂,孤食,貧困対策","前向き・推進意向","00:12:30〜00:15:00"

━━━━━━━━━━━━━━
【禁止事項】

* 空行やヘッダー以外の複数行出力
* 推測や脚色、主観的評価
* 挨拶・形式的な文言（例:「よろしくお願いします」「賛成多数で可決」）

以上。

    
    以下が対象テキストです：
    Timestamp: {timestamp}
    
    {text_chunk}
    """.strip()

    response = openai.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role": "user", "content": prompt_input}],
        max_tokens=2000,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def unify_and_save_csv(csv_texts, output_csv_path):
    """
    csv_texts: list of CSV strings from AI
    output_csv_path: path to the unified CSV
    This function concatenates multiple CSV strings (with identical headers) into one,
    ensuring the final CSV is a horizontal format where each line is one record.
    """
    import csv
    from io import StringIO
    import re

    known_header = [
        "headline",
        "overview",
        "category",
        "tags",
        "stance",
        "timestamp"
    ]

    all_rows = []

    for structured_text in csv_texts:
        row = parse_structured_output(structured_text)
        if row[0] == "NULL" or row[1] == "NULL":
            continue
        # 正規化
        row[4] = normalize_stance(row[4])
        all_rows.append(row)

    # Write to output CSV
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(known_header)

        def ensure_milliseconds(time_str):
            if not time_str:
                return time_str
            return time_str + ",000" if "," not in time_str else time_str

        for row in all_rows:
            if "〜" in row[5]:
                parts = row[5].split("〜", 1)
                row[5] = f"{ensure_milliseconds(parts[0].strip())}〜{ensure_milliseconds(parts[1].strip())}"
            else:
                row[5] = ensure_milliseconds(row[5].strip())
            writer.writerow(row)

def add_id_column_to_csv(csv_path, base_id):
    """
    Adds an ID column to the existing CSV file at csv_path.
    The ID will be the base_id provided.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.insert(0, 'id', base_id)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    ##########################
    # 以下、ユーザーのオリジナルプロンプトをそのまま
    ##########################
    system_prompt = """
あなたは議事録の要約に特化したAIです。次の要件を厳守してください。

【目的】
自治体関係者・民間事業者が「自分たちに関係ある話題かどうか」をヘッドラインと要約だけで判断できるようにすること。

【出力形式（CSV形式）】
1. ヘッダーは以下の6列：
   headline,overview,category,tags,stance,timestamp

2. 各列の内容：
- headline（60文字以内）:
  テーマの概要がすぐ分かるように簡潔に。

- overview（700文字以上）:
  以下を網羅：
  ・背景（どんな問題意識があるのか）
  ・誰が何を提案・指摘・質問し、誰がどう答えたか
  ・今後の方向性（導入するのか、検討中なのか、否定されたのか）
  ・議論の中心や結論が述べられた時間帯（例: (00:23:45)）を文中に挿入

- category:
  以下の20分類から該当するものを1つ記入
子育て・保育
学校教育・生涯学習
福祉・包摂（高齢・障がい・困窮）
医療・公衆衛生
防災・危機管理・安全安心
環境・エネルギー
経済・雇用・産業振興
農林水産業
都市整備・土地利用（ハード）
インフラ・公共施設
交通・モビリティ
デジタル・ICT推進
行政手続・窓口サービス
財政・税務
総務・人事・組織運営
政策立案・企画・計画
議会・選挙・ガバナンス
地域活性・コミュニティ（ソフト）
市民協働・広報
人権・男女共同参画（ダイバーシティ）
- tags:
  議論が活発だったものだけ、最大3つまでの重要キーワード（例: 高齢化,こども食堂）。
  特に頻出していないテーマは「NULL」とする。

- stance:
  以下のいずれかを記入（日本語）
    導入決定 / 導入済み / 内部決定・制度化 / 前向き / 検討中 / 調査・情報収集段階 / 慎重・消極的 / 反対・否定 / 情報不足・判断不能 / 実証・試行段階 / 要望・提案段階

【禁止事項】
・冗長なあいさつ、定型表現（例：「よろしくお願いします」「賛成多数で可決」など）は含めない。
・推測、個人的な解釈は禁止。テキストに明記された事実だけをもとにする。
・文字数が不足する場合は、もっと具体的なやり取り・発言内容を補足すること。

【出力例】
headline,overview,category,tags,stance,timestamp
"子ども食堂への支援拡大案",
"(00:12:30) 地域の子ども食堂への財政支援の必要性について議論。山田議員が孤食問題と経済格差の影響を挙げて支援拡大を提案。市側は現状の支援策を説明しつつ、他自治体の事例も参考にして柔軟に対応していきたいと回答。(00:14:50) 市長は『予算調整が必要だが、前向きに検討したい』と発言。複数議員から利用者数の把握と事後評価の必要性が指摘された。今後、令和7年度予算編成の中で具体化を目指す。",
"社会保障・福祉",
"子ども食堂,孤食,貧困対策",
"前向き",
"00:12:30〜00:15:00"
    """

    api_dir = "/Users/minkoil/yoyaku"
    
    # すでに完成フォルダーにCSVが存在するファイルは処理対象から除外する
    completed_dir = os.path.join(api_dir, "完成フォルダー")
    os.makedirs(completed_dir, exist_ok=True)
    completed_csv_ids = {os.path.splitext(f)[0] for f in os.listdir(completed_dir) if f.endswith(".csv")}

    done_srt_dir = os.path.join(api_dir, "done_srt")
    os.makedirs(done_srt_dir, exist_ok=True)

    print("=== DEBUG: Checking files in:", api_dir)
    print(os.listdir(api_dir))

    # ディレクトリ内の .srt ファイルをすべて読み込んで要約
    for filename in sorted(os.listdir(api_dir)):
        if not filename.endswith(".srt"):
            continue
        base_id = os.path.splitext(filename)[0]
        if base_id in completed_csv_ids:
            print(f"=== Skipping {filename} (already summarized) ===")
            continue

        file_path = os.path.join(api_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # 拡張子を除いたファイル名をIDとして活用（例: "abc.srt" → "abc"）
        # SRTファイルから (index, text) リストを生成
        print(f"=== Processing: {filename} ===")
        subs_list = parse_srt_as_subs(file_path)

        # 60分単位でチャンク化（先頭字幕のtimestampを保持）
        chunks = chunk_srt_subs_with_timestamp(subs_list, 60)

        all_csv_outputs = []

        for chunk_data in chunks:
            text_chunk = chunk_data['text']
            timestamp = chunk_data['timestamp']
 
            # Always do summary
            try:
                result = summarize_chunk(text_chunk, system_prompt, timestamp)
                all_csv_outputs.append(result)
            except Exception as e:
                print(f"=== ERROR summarizing chunk at {timestamp}: {e}")
                continue
 
        print("== Summarized chunk at", timestamp)
        print(result)
        print()
 
        # チャンク全体で中身が4行未満ならスキップ
        combined_text = "\n".join(all_csv_outputs).strip()
        line_count = len([line for line in combined_text.splitlines() if line.strip()])
        if line_count < 4:
            print(f"=== Skipping {base_id}: only {line_count} line(s) of output ===")
            # 再処理防止のため .srt ファイルも done_srt に移動
            shutil.move(file_path, os.path.join(done_srt_dir, filename))
            continue

        # すべてのチャンクのCSVを一つにまとめて保存
        csv_path = os.path.join(api_dir, f"{base_id}.csv")
        unify_and_save_csv(all_csv_outputs, csv_path)
        print(f"CSV saved to: {csv_path}")

        add_id_column_to_csv(csv_path, base_id)
        print(f"=== DEBUG: Appended ID column to CSV for base ID: {base_id} ===")
        
        # 元のSRTファイルも done_srt ディレクトリに移動（再処理防止のため）
        shutil.move(file_path, os.path.join(done_srt_dir, filename))
