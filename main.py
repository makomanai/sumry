# main.py
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
# 字幕を 2000文字単位でチャンク化しつつ、
# 先頭のインデックスを "timestamp" として保持
############################################
def chunk_srt_subs_with_timestamp(subs_list, chunk_size=2000):
    """
    subs_list: [(index, text), (index, text), ...]
    連結しながら 2000文字ごとにチャンクに分割。

    戻り値はリスト: [ { 'text': str, 'timestamp': str }, ... ]
    * 'timestamp' には チャンク内で最初に登場した字幕の timestamp を入れる
    """
    chunks = []
    current_text = []
    current_length = 0
    first_timestamp_in_chunk = None

    for (sub_index, start_time_str, sub_text) in subs_list:
        block_str = f"{start_time_str} {sub_text}\n"
        block_len = len(block_str)

        # 新しいチャンクを開始するときに first_timestamp_in_chunk をセット
        if first_timestamp_in_chunk is None:
            first_timestamp_in_chunk = start_time_str

        if current_length + block_len > chunk_size and current_length > 0:
            # 今のチャンクを確定
            chunks.append({
                'text': "".join(current_text),
                'timestamp': first_timestamp_in_chunk
            })
            # 新チャンクを開始
            current_text = []
            current_length = 0
            first_timestamp_in_chunk = start_time_str

        current_text.append(block_str)
        current_length += block_len

    # 最後の塊
    if current_text:
        chunks.append({
            'text': "".join(current_text),
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
    あなたは議事録の要約に特化したAIです。次の要件を厳守してください。
    
    【目的】
    自治体関係者・民間事業者が「自分たちに関係ある話題かどうか」をヘッドラインと要約だけで判断できるようにすること。
    
    【出力形式】
    以下の6項目を順番通り、【項目名】とその内容をセットで1行ずつ出力してください（CSV形式にはしないでください）。
    
    【headline】60文字以内で、議論の概要がすぐ分かる短いタイトル
    【overview】700文字以上で、以下を網羅：
     - 背景（どんな問題意識があるのか）
     - 誰が何を提案・指摘・質問し、誰がどう答えたか
     - 今後の方向性（導入するのか、検討中なのか、否定されたのか）
     - 議論の中心や結論が述べられた時間帯（例: (00:23:45)）を文中に必ず含めること
    【category】以下から該当するものを1つ以上（/区切りで複数可）：
    地域振興・活性化 / 社会保障・福祉 / 防災・安全 / 教育 / 環境・エネルギー / 医療・健康 / 都市計画・インフラ / 行政運営・ガバナンス / 文化・スポーツ / その他
    【tags】活発だった議論を示す最大3つのキーワード（ない場合は NULL）
    【stance】以下のいずれかを1つ記入（日本語）：
     導入決定
     導入済み
     内部決定・制度化
     前向き
     検討中
     調査・情報収集段階
     慎重・消極的
     反対・否定
     情報不足・判断不能
     実証・試行段階
     要望・提案段階
    【timestamp】議論の結論や方向性が示された時間帯（例: 00:20:00〜00:25:00）
    
    以下が対象テキストです：
    Timestamp: {timestamp}
    
    {text_chunk}
    """.strip()

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
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
  以下の10分類から該当するものを1つまたは複数記入（「/」区切り）
    地域振興・活性化 / 社会保障・福祉 / 防災・安全 / 教育 /
    環境・エネルギー / 医療・健康 / 都市計画・インフラ /
    行政運営・ガバナンス / 文化・スポーツ / その他

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

        # 2000文字単位でチャンク化（先頭字幕のtimestampを保持）
        chunks = chunk_srt_subs_with_timestamp(subs_list, 2000)

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
