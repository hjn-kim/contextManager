"""
Annotation type frequency & ambiguity checker.

Usage:
  python check_types.py                  # all splits
  python check_types.py --split eval     # single split
  python check_types.py --ambiguous      # show ambiguous cases only
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

ALL_TYPES = ["agenda_item", "detail", "medication", "social_history",
             "follow_up", "question", "question_unanswered"]


def analyze(convs: list[dict]) -> dict:
    type_counter = Counter()
    multi_type = []      # utterances with multiple types in detail_list
    ambiguous = []       # potential edge cases worth reviewing

    for conv in convs:
        utt_map = {u["id"]: u["text"] for u in conv["utterances"]}
        for ann in conv.get("annotations", []):
            detail_list = ann.get("detail_list") or []

            types_in_utt = [e["type"] for e in detail_list]
            type_counter.update(types_in_utt)

            if len(detail_list) > 1:
                multi_type.append({
                    "conv_id": conv["id"],
                    "utterance_id": ann["utterance_id"],
                    "text": utt_map.get(ann["utterance_id"], ""),
                    "types": types_in_utt,
                })

            for entry in detail_list:
                text = utt_map.get(ann["utterance_id"], "").lower()
                t = entry["type"]

                reasons = []
                # "Do/Are/Is there any questions?" → agenda_item or question 경계
                if any(kw in text for kw in ["any questions", "any question"]):
                    reasons.append("'any questions?' 포함")
                # question인데 답변이 없을 수 있음
                if t == "question" and "?" not in text:
                    reasons.append("question인데 물음표 없음")
                # social_history vs detail 경계
                if t == "detail" and any(kw in text for kw in ["smoke", "drink", "alcohol", "exercise", "occupation", "live"]):
                    reasons.append("social_history일 수 있음 (생활습관 키워드)")
                # agenda_item인데 브리핑 발화일 수 있음
                if t == "agenda_item" and any(kw in text for kw in ["year-old", "years old", "history of", "presents for"]):
                    reasons.append("브리핑 발화 → social_history여야 할 수 있음")

                if reasons:
                    ambiguous.append({
                        "conv_id": conv["id"],
                        "utterance_id": ann["utterance_id"],
                        "type": t,
                        "text": utt_map.get(ann["utterance_id"], "")[:120],
                        "reasons": reasons,
                    })

    return {
        "total_annotations": sum(type_counter.values()),
        "total_convs": len(convs),
        "type_counter": type_counter,
        "multi_type": multi_type,
        "ambiguous": ambiguous,
    }


def print_report(split: str, result: dict, show_ambiguous: bool):
    print(f"\n{'='*60}")
    print(f"  Split: {split}  ({result['total_convs']} conversations)")
    print(f"{'='*60}")

    print(f"\n[Type 빈도]  총 {result['total_annotations']}개 annotation\n")
    total = result["total_annotations"]
    for t in ALL_TYPES:
        n = result["type_counter"].get(t, 0)
        bar = "#" * int(n / max(total, 1) * 40)
        print(f"  {t:<22} {n:>4}  ({n/max(total,1)*100:5.1f}%)  {bar}")

    others = {k: v for k, v in result["type_counter"].items() if k not in ALL_TYPES}
    if others:
        print(f"\n  [알 수 없는 type]")
        for k, v in others.items():
            print(f"  {k:<22} {v:>4}")

    print(f"\n[다중 type 발화]  {len(result['multi_type'])}개")
    for item in result["multi_type"][:10]:
        print(f"  {item['conv_id']} {item['utterance_id']}  {item['types']}")
        print(f"    \"{item['text'][:100]}\"")
    if len(result["multi_type"]) > 10:
        print(f"  ... 외 {len(result['multi_type'])-10}개")

    if show_ambiguous:
        print(f"\n[애매한 케이스]  {len(result['ambiguous'])}개")
        for item in result["ambiguous"]:
            print(f"  {item['conv_id']} {item['utterance_id']}  [{item['type']}]")
            print(f"    사유: {', '.join(item['reasons'])}")
            print(f"    \"{item['text']}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="분석할 JSON 파일이나 폴더 경로 (예: ..\\data\\aci 또는 ..\\data\\aci\\D2N001.json)")
    parser.add_argument("--ambiguous", action="store_true", help="애매한 케이스 출력")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} 경로가 존재하지 않습니다.")
        return
    
    convs = []
    if path.is_file():
        convs.append(json.load(open(path, encoding="utf-8")))
    elif path.is_dir():
        for f in sorted(path.glob("*.json")):
            convs.append(json.load(open(f, encoding="utf-8")))
    else:
        print(f"Error: {path} 은 올바른 파일이나 디렉토리가 아닙니다.")
        return
        
    result = analyze(convs)
    print_report(path.name, result, args.ambiguous)


if __name__ == "__main__":
    main()
